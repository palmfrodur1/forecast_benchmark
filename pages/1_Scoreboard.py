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
def _get_item_features_item_ids(project: str) -> set[str]:
    """Return item_ids present in item_features for a project.

    Used for optional UI filtering. Returns empty set if table missing/empty.
    """

    if _use_mysql():
        try:
            df = _mysql_read_df(
                "SELECT item_id FROM item_features WHERE project = :project",
                {"project": project},
            )
        except Exception:
            return set()
    else:
        con = get_connection()
        try:
            df = con.execute(
                "SELECT item_id FROM item_features WHERE project = ?",
                [project],
            ).fetchdf()
        except Exception:
            return set()
        finally:
            con.close()

    if df is None or df.empty or 'item_id' not in df.columns:
        return set()
    return set(df['item_id'].dropna().astype(str).tolist())


@st.cache_data(ttl=60)
def _get_item_name_map(project: str) -> dict[str, str]:
    """Return {item_id: name} from item_features for a project.

    Missing/NULL names are excluded so items not present (or with null name)
    will display as blank.
    """

    if _use_mysql():
        try:
            df = _mysql_read_df(
                "SELECT item_id, name FROM item_features WHERE project = :project",
                {"project": project},
            )
        except Exception:
            return {}
    else:
        con = get_connection()
        try:
            df = con.execute(
                "SELECT item_id, name FROM item_features WHERE project = ?",
                [project],
            ).fetchdf()
        except Exception:
            return {}
        finally:
            con.close()

    if df is None or df.empty or 'item_id' not in df.columns or 'name' not in df.columns:
        return {}

    df = df.copy()
    df['item_id'] = df['item_id'].astype(str)
    df = df[df['name'].notna()].copy()
    if df.empty:
        return {}
    df['name'] = df['name'].astype(str)
    return dict(zip(df['item_id'], df['name']))


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


def _ensure_scoreboard_periods_table() -> None:
    if _use_mysql():
        eng = _mysql_engine()
        with eng.begin() as conn:
            conn.execute(
                sa.text(
                    """
                    CREATE TABLE IF NOT EXISTS scoreboard_periods (
                        project              VARCHAR(255) NOT NULL,
                        period_key           VARCHAR(64)  NOT NULL,
                        start_date           DATE         NOT NULL,
                        end_date             DATE         NOT NULL,
                        abs_actual_threshold DOUBLE       NULL,
                        last_updated         TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE KEY uq_scoreboard_periods (project, period_key)
                    )
                    """
                )
            )
        return

    con = get_connection()
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS scoreboard_periods (
                project              VARCHAR,
                period_key           VARCHAR,
                start_date           DATE,
                end_date             DATE,
                abs_actual_threshold DOUBLE,
                last_updated         TIMESTAMP DEFAULT current_timestamp
            );
            """
        )
    finally:
        con.close()


def _load_saved_period(project: str, *, period_key: str = "current") -> tuple[date, date, float | None] | None:
    _ensure_scoreboard_periods_table()

    if _use_mysql():
        df = _mysql_read_df(
            """
            SELECT start_date, end_date, abs_actual_threshold
            FROM scoreboard_periods
            WHERE project = :project AND period_key = :period_key
            ORDER BY last_updated DESC
            LIMIT 1
            """,
            {"project": project, "period_key": period_key},
        )
    else:
        con = get_connection()
        try:
            df = con.execute(
                """
                SELECT start_date, end_date, abs_actual_threshold
                FROM scoreboard_periods
                WHERE project = ? AND period_key = ?
                ORDER BY last_updated DESC
                LIMIT 1
                """,
                [project, period_key],
            ).fetchdf()
        finally:
            con.close()

    if df.empty:
        return None

    start = pd.to_datetime(df.iloc[0]["start_date"]).date()
    end = pd.to_datetime(df.iloc[0]["end_date"]).date()
    thr_raw = df.iloc[0].get("abs_actual_threshold")
    thr = None if pd.isna(thr_raw) else float(thr_raw)
    return start, end, thr


def _save_period(
    *,
    project: str,
    start_date: date,
    end_date: date,
    abs_actual_threshold: float | None,
    period_key: str = "current",
) -> None:
    _ensure_scoreboard_periods_table()

    if _use_mysql():
        eng = _mysql_engine()
        with eng.begin() as conn:
            conn.execute(
                sa.text(
                    """
                    INSERT INTO scoreboard_periods (project, period_key, start_date, end_date, abs_actual_threshold)
                    VALUES (:project, :period_key, :start_date, :end_date, :abs_actual_threshold)
                    ON DUPLICATE KEY UPDATE
                        start_date = VALUES(start_date),
                        end_date = VALUES(end_date),
                        abs_actual_threshold = VALUES(abs_actual_threshold),
                        last_updated = CURRENT_TIMESTAMP
                    """
                ),
                {
                    "project": project,
                    "period_key": period_key,
                    "start_date": start_date,
                    "end_date": end_date,
                    "abs_actual_threshold": abs_actual_threshold,
                },
            )
        return

    con = get_connection()
    try:
        con.execute("DELETE FROM scoreboard_periods WHERE project = ? AND period_key = ?", [project, period_key])
        con.execute(
            """
            INSERT INTO scoreboard_periods (project, period_key, start_date, end_date, abs_actual_threshold)
            VALUES (?, ?, ?, ?, ?)
            """,
            [project, period_key, start_date, end_date, abs_actual_threshold],
        )
    finally:
        con.close()


def _clamp_date(d: date, lo: date, hi: date) -> date:
    if d < lo:
        return lo
    if d > hi:
        return hi
    return d


def _pivot_grid(df: pd.DataFrame, metric_names: list[str], methods: list[str] | None) -> pd.DataFrame:
    """Return grid with rows = (item_id, forecast_method) and columns = metrics."""

    d = df[df["metric_name"].isin(metric_names)].copy()
    if methods:
        d = d[d["forecast_method"].isin(methods)].copy()

    if d.empty:
        return pd.DataFrame()

    pivot = d.pivot_table(
        index=["item_id", "forecast_method"],
        columns="metric_name",
        values="metric_value",
        aggfunc="first",
    ).reset_index()

    base_cols = ["item_id", "forecast_method"]
    metric_cols = [c for c in metric_names if c in pivot.columns]
    # Include any unexpected metric columns too (stable ordering).
    other_metric_cols = sorted([c for c in pivot.columns if c not in base_cols + metric_cols])
    return pivot[base_cols + metric_cols + other_metric_cols]


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

    # Keep date widgets valid across app restarts and project switches.
    # Streamlit widgets will error if the provided default lies outside min/max.
    if min_date is not None and max_date is not None:
        saved_period = _load_saved_period(project)
        saved_start: date | None = None
        saved_end: date | None = None
        saved_thr: float | None = None
        if saved_period is not None:
            saved_start, saved_end, saved_thr = saved_period

        # Clamp DB-loaded defaults into project bounds.
        start_default = _clamp_date(saved_start, min_date, max_date) if saved_start is not None else min_date
        end_default = _clamp_date(saved_end, min_date, max_date) if saved_end is not None else max_date
        if start_default > end_default:
            start_default, end_default = min_date, max_date
        thr_default = float(saved_thr) if saved_thr is not None else 1.0

        # On project switch, clear widget state so the new project's defaults can take effect.
        last_project = st.session_state.get('score_last_project')
        if last_project != project:
            st.session_state['score_last_project'] = project
            st.session_state.pop('score_start_date', None)
            st.session_state.pop('score_end_date', None)
            st.session_state.pop('score_abs_actual_threshold', None)

        # If a persisted widget state is out of bounds (e.g. data changed), clear it.
        cur_start = st.session_state.get('score_start_date')
        if isinstance(cur_start, date) and (cur_start < min_date or cur_start > max_date):
            st.session_state.pop('score_start_date', None)
        cur_end = st.session_state.get('score_end_date')
        if isinstance(cur_end, date) and (cur_end < min_date or cur_end > max_date):
            st.session_state.pop('score_end_date', None)

        cur_start = st.session_state.get('score_start_date')
        cur_end = st.session_state.get('score_end_date')
        if isinstance(cur_start, date) and isinstance(cur_end, date) and cur_start > cur_end:
            st.session_state.pop('score_start_date', None)
            st.session_state.pop('score_end_date', None)

    st.subheader("Metrics")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        if min_date is None or max_date is None:
            start_date = st.date_input("Start date", value=date.today(), disabled=True, key='score_start_date')
        else:
            start_date = st.date_input(
                "Start date",
                value=start_default,
                min_value=min_date,
                max_value=max_date,
                key='score_start_date',
            )
    with c2:
        if min_date is None or max_date is None:
            end_date = st.date_input("End date", value=date.today(), disabled=True, key='score_end_date')
        else:
            end_date = st.date_input(
                "End date",
                value=end_default,
                min_value=min_date,
                max_value=max_date,
                key='score_end_date',
            )
    with c3:
        abs_actual_threshold = st.number_input(
            "Min abs actual",
            min_value=0.0,
            value=thr_default,
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
                _save_period(
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

    metric_options = ["All"] + metric_names
    if 'score_metrics' in st.session_state:
        prev = st.session_state.get('score_metrics')
        if isinstance(prev, (list, tuple)):
            st.session_state['score_metrics'] = [m for m in prev if m in metric_options]
    if 'score_methods' in st.session_state:
        prev = st.session_state.get('score_methods')
        if isinstance(prev, (list, tuple)):
            st.session_state['score_methods'] = [m for m in prev if m in methods_all]

    with st.sidebar:
        selected_metrics_raw = st.multiselect(
            "Metrics",
            metric_options,
            default=["All"],
            key='score_metrics',
            help="Select one or more metrics. Choose 'All' for everything.",
        )
        methods = st.multiselect("Forecast methods", methods_all, default=methods_all, key='score_methods')
        only_item_features = st.checkbox(
            "Only items in item_features",
            value=False,
            key='score_only_item_features',
            help="When enabled, only show items that exist in the item_features table for this project.",
        )
        item_filter = st.text_input("Item filter (substring)", value="", key='score_item_filter')
        top_n = st.number_input("Max items", min_value=10, max_value=20000, value=500, step=10, key='score_top_n')
        sort_mode = st.selectbox(
            "Sort rows by",
            options=["Best (min across selected metrics)", "Worst (max across selected metrics)", "Item id"],
            index=0,
            key='score_sort_mode',
        )

    if (not selected_metrics_raw) or ("All" in selected_metrics_raw):
        selected_metrics = metric_names
    else:
        selected_metrics = [m for m in selected_metrics_raw if m != "All"]
        if not selected_metrics:
            selected_metrics = metric_names

    grid = _pivot_grid(df, metric_names=selected_metrics, methods=methods)
    if grid.empty:
        st.info("No rows match your filters.")
        return

    # Add item name (blank if missing in item_features).
    name_map = _get_item_name_map(project)
    grid = grid.copy()
    grid.insert(1, 'name', grid['item_id'].astype(str).map(name_map).fillna(''))

    if only_item_features:
        allowed = _get_item_features_item_ids(project)
        if not allowed:
            st.info("No rows found in item_features for this project.")
            return
        grid = grid[grid["item_id"].astype(str).isin(allowed)]
        if grid.empty:
            st.info("No rows match your filters.")
            return

    if item_filter.strip():
        s = item_filter.strip().lower()
        grid = grid[grid["item_id"].astype(str).str.lower().str.contains(s)]

    value_cols = [c for c in grid.columns if c not in ("item_id", "name", "forecast_method")]

    if sort_mode == "Best (min across selected metrics)":
        if value_cols:
            sort_key = grid[value_cols].min(axis=1, skipna=True)
            grid = grid.loc[sort_key.sort_values(ascending=True).index]
    elif sort_mode == "Worst (max across selected metrics)":
        if value_cols:
            sort_key = grid[value_cols].max(axis=1, skipna=True)
            grid = grid.loc[sort_key.sort_values(ascending=False).index]
    else:
        grid = grid.sort_values(["item_id", "forecast_method"], ascending=[True, True])

    grid = grid.head(int(top_n))

    metric_label = "All metrics" if len(selected_metrics) == len(metric_names) else ", ".join(selected_metrics)
    st.subheader(f"{project} — {metric_label}")
    fmt = {c: "{:.2f}" for c in value_cols}
    st.dataframe(grid.style.format(fmt), use_container_width=True)

    with st.expander("Raw metric rows"):
        d = df.copy()
        d = d[d["metric_name"].isin(selected_metrics)].copy()
        if methods:
            d = d[d["forecast_method"].isin(methods)].copy()
        if only_item_features:
            allowed = _get_item_features_item_ids(project)
            if allowed:
                d = d[d["item_id"].astype(str).isin(allowed)].copy()
        if item_filter.strip():
            s = item_filter.strip().lower()
            d = d[d["item_id"].str.lower().str.contains(s)].copy()
        st.dataframe(d, use_container_width=True)


if __name__ == "__main__":
    main()
