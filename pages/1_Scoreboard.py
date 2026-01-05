import os
import pandas as pd
import streamlit as st

from datetime import date

import sqlalchemy as sa

from db import get_connection, get_mysql_engine
from metrics import recompute_all_metrics, recompute_all_metrics_mysql

from nav import render_sidebar_nav


st.set_page_config(page_title="Scoreboard", layout="wide")

render_sidebar_nav()


def _use_mysql() -> bool:
    # When the user runs `streamlit run app_mysql.py`, we want Scoreboard to
    # read from MySQL too (not DuckDB).
    if st.session_state.get("_entrypoint") == "app_mysql.py":
        return True
    return (os.getenv("BENCHMARK_DB_BACKEND") or "").lower() == "mysql"


@st.cache_resource
def _mysql_engine() -> sa.Engine:
    return get_mysql_engine()


def _mysql_read_df(sql: str, params: dict | None = None) -> pd.DataFrame:
    eng = _mysql_engine()
    with eng.connect() as conn:
        return pd.read_sql(sa.text(sql), conn, params=params or {})


@st.cache_data(ttl=60)
def _get_projects() -> list[str]:
    if _use_mysql():
        df = _mysql_read_df("SELECT DISTINCT project FROM sales_history ORDER BY project")
    else:
        con = get_connection()
        try:
            df = con.execute(
                "SELECT DISTINCT project FROM sales_history ORDER BY project"
            ).fetchdf()
        finally:
            con.close()
    return df["project"].dropna().astype(str).tolist()


@st.cache_data(ttl=60)
def _load_score_rows(project: str) -> pd.DataFrame:
    """Load latest metrics per (item, base_method, metric_name).

    - Collapses tagged runs like method@YYYY... to base_method=method
    - Prefers untagged rows if present; otherwise chooses max(tag) as latest
    """

    if _use_mysql():
        df = _mysql_read_df(
            """
            WITH ranked AS (
                SELECT
                    project,
                    item_id,
                    SUBSTRING_INDEX(forecast_method, '@', 1) AS base_method,
                    forecast_method,
                    metric_name,
                    metric_value,
                    n_points,
                    CASE WHEN LOCATE('@', forecast_method) = 0 THEN 1 ELSE 0 END AS is_untagged
                FROM forecast_metrics
                WHERE project = :project
            ),
            chosen AS (
                SELECT
                    project,
                    item_id,
                    base_method,
                    metric_name,
                    COALESCE(
                        MAX(CASE WHEN is_untagged = 1 THEN forecast_method END),
                        MAX(forecast_method)
                    ) AS chosen_method
                FROM ranked
                GROUP BY project, item_id, base_method, metric_name
            )
            SELECT
                r.item_id,
                r.base_method AS forecast_method,
                r.metric_name,
                r.metric_value,
                r.n_points
            FROM ranked r
            JOIN chosen c
              ON r.project = c.project
             AND r.item_id = c.item_id
             AND r.base_method = c.base_method
             AND r.metric_name = c.metric_name
             AND r.forecast_method = c.chosen_method
            ORDER BY r.item_id, r.base_method, r.metric_name
            """,
            {"project": project},
        )
    else:
        con = get_connection()
        try:
            df = con.execute(
                """
                WITH ranked AS (
                    SELECT
                        project,
                        item_id,
                        split_part(forecast_method, '@', 1) AS base_method,
                        forecast_method,
                        metric_name,
                        metric_value,
                        n_points,
                        CASE WHEN strpos(forecast_method, '@') = 0 THEN 1 ELSE 0 END AS is_untagged
                    FROM forecast_metrics
                    WHERE project = ?
                ),
                chosen AS (
                    SELECT
                        project,
                        item_id,
                        base_method,
                        metric_name,
                        COALESCE(
                            MAX(CASE WHEN is_untagged = 1 THEN forecast_method END),
                            MAX(forecast_method)
                        ) AS chosen_method
                    FROM ranked
                    GROUP BY 1,2,3,4
                )
                SELECT
                    r.item_id,
                    r.base_method AS forecast_method,
                    r.metric_name,
                    r.metric_value,
                    r.n_points
                FROM ranked r
                JOIN chosen c
                  ON r.project = c.project
                 AND r.item_id = c.item_id
                 AND r.base_method = c.base_method
                 AND r.metric_name = c.metric_name
                 AND r.forecast_method = c.chosen_method
                ORDER BY r.item_id, r.base_method, r.metric_name
                """,
                [project],
            ).fetchdf()
        finally:
            con.close()

    if df.empty:
        return pd.DataFrame(columns=["item_id", "forecast_method", "metric_name", "metric_value", "n_points"])

    df["item_id"] = df["item_id"].astype(str)
    df["forecast_method"] = df["forecast_method"].astype(str)
    df["metric_name"] = df["metric_name"].astype(str)
    df["metric_value"] = pd.to_numeric(df["metric_value"], errors="coerce")
    df["n_points"] = pd.to_numeric(df["n_points"], errors="coerce")
    return df


@st.cache_data(ttl=60)
def _get_project_date_bounds(project: str) -> tuple[date | None, date | None]:
    if _use_mysql():
        df = _mysql_read_df(
            "SELECT MIN(sale_date) AS min_date, MAX(sale_date) AS max_date FROM sales_history WHERE project = :project",
            {"project": project},
        )
    else:
        con = get_connection()
        try:
            df = con.execute(
                "SELECT MIN(sale_date) AS min_date, MAX(sale_date) AS max_date FROM sales_history WHERE project = ?",
                [project],
            ).fetchdf()
        finally:
            con.close()
    if df.empty or df.iloc[0].isna().any():
        return None, None
    return pd.to_datetime(df.iloc[0]["min_date"]).date(), pd.to_datetime(df.iloc[0]["max_date"]).date()


def _pivot_grid(df: pd.DataFrame, metric_name: str, methods: list[str] | None) -> pd.DataFrame:
    d = df[df["metric_name"] == metric_name].copy()
    if methods:
        d = d[d["forecast_method"].isin(methods)].copy()

    if d.empty:
        return pd.DataFrame()

    pivot = d.pivot(index="item_id", columns="forecast_method", values="metric_value")
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    return pivot


def main() -> None:
    st.title("Scoreboard")
    st.caption("Compare forecast quality across items and methods.")

    projects = _get_projects()
    if not projects:
        st.warning("No projects found in sales history.")
        return

    with st.sidebar:
        st.header("Scoreboard")
        if 'active_project' in st.session_state and st.session_state['active_project'] in projects:
            if 'score_project' not in st.session_state or st.session_state.get('score_project') != st.session_state.get('active_project'):
                st.session_state['score_project'] = st.session_state['active_project']
        elif 'score_project' in st.session_state and st.session_state['score_project'] in projects:
            st.session_state['active_project'] = st.session_state['score_project']
        else:
            st.session_state['active_project'] = projects[0]
            st.session_state['score_project'] = projects[0]

        def _on_score_project_change() -> None:
            st.session_state['active_project'] = st.session_state.get('score_project')

        project = st.selectbox(
            "Project",
            projects,
            index=0,
            key='score_project',
            on_change=_on_score_project_change,
        )

    min_date, max_date = _get_project_date_bounds(project)

    st.subheader("Metrics")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        if min_date is None or max_date is None:
            start_date = st.date_input("Start date", value=date.today(), disabled=True)
        else:
            start_date = st.date_input(
                "Start date",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
            )
    with c2:
        if min_date is None or max_date is None:
            end_date = st.date_input("End date", value=date.today(), disabled=True)
        else:
            end_date = st.date_input(
                "End date",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
            )
    with c3:
        abs_actual_threshold = st.number_input(
            "Min abs actual",
            min_value=0.0,
            value=1.0,
            step=1.0,
            key='score_abs_actual_threshold',
            help="Points with abs(actual) < threshold are ignored; negative actuals are treated as 0.",
        )
    with c4:
        recompute_clicked = st.button("Calculate metrics", width="stretch")

    if recompute_clicked:
        if min_date is None or max_date is None:
            st.error("No sales history dates found for this project.")
            return
        if start_date > end_date:
            st.error("Start date must be <= end date.")
            return
        with st.spinner("Computing metrics for the selected period..."):
            try:
                if _use_mysql():
                    recompute_all_metrics_mysql(
                        project=project,
                        start_date=start_date,
                        end_date=end_date,
                        abs_actual_threshold=float(abs_actual_threshold),
                    )
                else:
                    recompute_all_metrics(
                        project=project,
                        start_date=start_date,
                        end_date=end_date,
                        abs_actual_threshold=float(abs_actual_threshold),
                    )
                st.cache_data.clear()
                st.success("Metrics computed.")
                st.rerun()
            except Exception as e:
                st.error(f"Error computing metrics: {e}")
                return

    df = _load_score_rows(project)
    if df.empty:
        st.info("No metrics found yet. Use the controls above or run `python3 metrics.py`.")
        return

    metric_names = sorted(df["metric_name"].dropna().unique().tolist())
    methods_all = sorted(df["forecast_method"].dropna().unique().tolist())

    if 'score_metric' in st.session_state and st.session_state['score_metric'] not in metric_names:
        st.session_state['score_metric'] = metric_names[0]
    if 'score_methods' in st.session_state:
        prev = st.session_state.get('score_methods')
        if isinstance(prev, (list, tuple)):
            st.session_state['score_methods'] = [m for m in prev if m in methods_all]

    with st.sidebar:
        metric = st.selectbox("Metric", metric_names, index=0, key='score_metric')
        methods = st.multiselect("Forecast methods", methods_all, default=methods_all, key='score_methods')
        item_filter = st.text_input("Item filter (substring)", value="", key='score_item_filter')
        top_n = st.number_input("Max items", min_value=10, max_value=20000, value=500, step=10, key='score_top_n')
        sort_mode = st.selectbox(
            "Sort rows by",
            options=["Best (min across selected methods)", "Worst (max across selected methods)", "Item id"],
            index=0,
            key='score_sort_mode',
        )

    grid = _pivot_grid(df, metric_name=metric, methods=methods)
    if grid.empty:
        st.info("No rows match your filters.")
        return

    if item_filter.strip():
        s = item_filter.strip().lower()
        grid = grid[grid.index.to_series().str.lower().str.contains(s)]

    if sort_mode == "Best (min across selected methods)":
        sort_key = grid.min(axis=1, skipna=True)
        grid = grid.loc[sort_key.sort_values(ascending=True).index]
    elif sort_mode == "Worst (max across selected methods)":
        sort_key = grid.max(axis=1, skipna=True)
        grid = grid.loc[sort_key.sort_values(ascending=False).index]
    else:
        grid = grid.sort_index()

    grid = grid.head(int(top_n))

    st.subheader(f"{project} — {metric}")
    st.dataframe(grid.style.format("{:.2f}"), use_container_width=True)

    with st.expander("Raw metric rows"):
        d = df.copy()
        if methods:
            d = d[d["forecast_method"].isin(methods)].copy()
        if item_filter.strip():
            s = item_filter.strip().lower()
            d = d[d["item_id"].str.lower().str.contains(s)].copy()
        st.dataframe(d, use_container_width=True)


if __name__ == "__main__":
    main()
