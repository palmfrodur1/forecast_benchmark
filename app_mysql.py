# app_mysql.py
# MySQL-backed Streamlit app for the forecast_benchmark project.

import os
from datetime import datetime, timedelta, date

import altair as alt
import pandas as pd
import streamlit as st
import sqlalchemy as sa

from db import get_mysql_engine, init_mysql_db
from nav import render_sidebar_nav


st.set_page_config(page_title="Forecast Benchmark (MySQL)")

# Used by nav.render_sidebar_nav() to link to the correct entrypoint file.
st.session_state['_entrypoint'] = 'app_mysql.py'

render_sidebar_nav()


# Ensure tables exist in MySQL (schema = MYSQL_DATABASE, default forecast_benchmark)
init_mysql_db()


@st.cache_resource
def _engine() -> sa.Engine:
    return get_mysql_engine()


def _read_df(sql: str, params: dict | None = None) -> pd.DataFrame:
    eng = _engine()
    with eng.connect() as conn:
        return pd.read_sql(sa.text(sql), conn, params=params or {})


def _in_params(name: str, values: list[str]) -> tuple[str, dict]:
    """Return (sql_fragment, params) for an IN (...) list using named params."""
    values = [str(v) for v in (values or [])]
    params: dict[str, object] = {}
    keys: list[str] = []
    for i, v in enumerate(values):
        k = f"{name}_{i}"
        keys.append(f":{k}")
        params[k] = v
    if not keys:
        return "(NULL)", {}
    return f"({', '.join(keys)})", params


def _aggregate_monthly_history_series(history_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate chart history series to monthly totals (month start dates)."""
    if history_df is None or history_df.empty:
        return pd.DataFrame(columns=["date", "value", "series"])

    df = history_df.copy()
    if "date" not in df.columns or "value" not in df.columns:
        return pd.DataFrame(columns=["date", "value", "series"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "value", "series"])

    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp(how="start")
    df = df.groupby(["date"], as_index=False)["value"].sum()
    df["series"] = "History (Monthly)"
    return df


@st.cache_data(ttl=60)
def get_projects() -> list[str]:
    df = _read_df("SELECT DISTINCT project FROM sales_history ORDER BY project")
    if df.empty:
        return []
    return df["project"].astype(str).tolist()


@st.cache_data(ttl=60)
def get_items(project: str) -> list[str]:
    df = _read_df(
        """
        SELECT DISTINCT item_id
        FROM sales_history
        WHERE project = :project
        ORDER BY item_id
        """,
        {"project": project},
    )
    if df.empty:
        return []
    return df["item_id"].astype(str).tolist()


@st.cache_data(ttl=60)
def get_forecast_methods(project: str, item_id: str) -> list[str]:
    df = _read_df(
        """
        SELECT DISTINCT SUBSTRING_INDEX(forecast_method, '@', 1) AS forecast_method
        FROM forecasts
        WHERE project = :project AND item_id = :item_id
        ORDER BY forecast_method
        """,
        {"project": project, "item_id": item_id},
    )
    if df.empty:
        return []
    return df["forecast_method"].astype(str).tolist()


@st.cache_data(ttl=60)
def load_series(project: str, item_id: str, methods: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    history = _read_df(
        """
        SELECT
            sale_date AS date,
            sales    AS value,
            'History' AS series
        FROM sales_history
        WHERE project = :project AND item_id = :item_id
        ORDER BY sale_date
        """,
        {"project": project, "item_id": item_id},
    )

    if methods:
        in_clause, in_params = _in_params("m", methods)
        sql = f"""
            WITH chosen AS (
                SELECT
                    SUBSTRING_INDEX(forecast_method, '@', 1) AS base_method,
                    COALESCE(
                        MAX(CASE WHEN LOCATE('@', forecast_method) = 0 THEN forecast_method END),
                        MAX(forecast_method)
                    ) AS chosen_method
                FROM forecasts
                WHERE project = :project
                  AND item_id = :item_id
                  AND SUBSTRING_INDEX(forecast_method, '@', 1) IN {in_clause}
                GROUP BY 1
            )
            SELECT
                f.forecast_date AS date,
                f.forecast      AS value,
                c.base_method   AS series
            FROM forecasts f
            JOIN chosen c
              ON f.forecast_method = c.chosen_method
            WHERE f.project = :project
              AND f.item_id = :item_id
            ORDER BY f.forecast_date
        """
        params = {"project": project, "item_id": item_id, **in_params}
        forecasts = _read_df(sql, params)
    else:
        forecasts = pd.DataFrame(columns=["date", "value", "series"])

    combined = pd.concat([history, forecasts], ignore_index=True)
    return history, forecasts, combined


@st.cache_data(ttl=60)
def load_metrics(project: str, item_id: str, methods: list[str]) -> pd.DataFrame:
    if not methods:
        return pd.DataFrame(columns=["forecast_method", "metric_name", "metric_value", "n_points"])

    in_clause, in_params = _in_params("m", methods)
    sql = f"""
        WITH chosen AS (
            SELECT
                SUBSTRING_INDEX(forecast_method, '@', 1) AS base_method,
                COALESCE(
                    MAX(CASE WHEN LOCATE('@', forecast_method) = 0 THEN forecast_method END),
                    MAX(forecast_method)
                ) AS chosen_method
            FROM forecast_metrics
            WHERE project = :project
              AND item_id = :item_id
              AND SUBSTRING_INDEX(forecast_method, '@', 1) IN {in_clause}
            GROUP BY 1
        )
        SELECT
            c.base_method AS forecast_method,
            m.metric_name,
            m.metric_value,
            m.n_points
        FROM forecast_metrics m
        JOIN chosen c
          ON m.forecast_method = c.chosen_method
        WHERE m.project = :project
          AND m.item_id = :item_id
        ORDER BY c.base_method, m.metric_name
    """

    params = {"project": project, "item_id": item_id, **in_params}
    return _read_df(sql, params)


@st.cache_data(ttl=60)
def get_history_date_range(project: str, item_id: str | None = None) -> tuple[date | None, date | None]:
    if item_id is None:
        df = _read_df(
            """
            SELECT MIN(sale_date) AS min_date, MAX(sale_date) AS max_date
            FROM sales_history
            WHERE project = :project
            """,
            {"project": project},
        )
    else:
        df = _read_df(
            """
            SELECT MIN(sale_date) AS min_date, MAX(sale_date) AS max_date
            FROM sales_history
            WHERE project = :project AND item_id = :item_id
            """,
            {"project": project, "item_id": item_id},
        )

    if df.empty or df.iloc[0].isna().any():
        return None, None
    return pd.to_datetime(df.iloc[0]["min_date"]).date(), pd.to_datetime(df.iloc[0]["max_date"]).date()


def main() -> None:
    st.title("Forecast Benchmark Explorer (MySQL)")

    projects = get_projects()
    if not projects:
        st.warning("No data found in MySQL. Load data into sales_history/forecasts first.")
        return

    if "active_project" not in st.session_state or st.session_state.get("active_project") not in projects:
        st.session_state["active_project"] = projects[0]

    project = st.sidebar.selectbox("Project", projects, key="mysql_project")

    items = get_items(project)
    if not items:
        st.warning("No items for this project.")
        return

    item_id = st.sidebar.selectbox("Item", items, key="mysql_item")

    methods = get_forecast_methods(project, item_id)
    selected_methods = methods  # show all by default

    history, forecasts, combined = load_series(project, item_id, selected_methods)
    metrics_df = load_metrics(project, item_id, selected_methods)

    st.subheader(f"Item: {item_id} — Project: {project}")

    st.markdown("---")

    col1, col2, col3 = st.columns([0.08, 0.76, 0.16])
    with col1:
        months_back = st.number_input("Months back", min_value=0, max_value=120, value=24, step=1, key="months_back")
    with col3:
        months_forward = st.number_input(
            "Months forward", min_value=0, max_value=120, value=12, step=1, key="months_forward"
        )

    today = datetime.now()
    first_of_current_month = datetime(today.year, today.month, 1)
    start_date = (first_of_current_month - timedelta(days=30 * int(months_back))).date()
    end_date = (first_of_current_month + timedelta(days=30 * int(months_forward) + 30)).date()

    show_monthly_history = st.checkbox(
        "Show monthly aggregated history",
        value=True,
        key="mysql_show_monthly_history",
        help="Overlays a monthly-summed history series dated to the 1st of each month.",
    )

    combined_for_chart = combined.copy()
    if show_monthly_history:
        monthly_hist = _aggregate_monthly_history_series(history)
        if not monthly_hist.empty:
            combined_for_chart = pd.concat([combined_for_chart, monthly_hist], ignore_index=True)

    combined_for_chart["date"] = pd.to_datetime(combined_for_chart["date"], errors="coerce")
    combined_for_chart = combined_for_chart[combined_for_chart["date"].notna()].copy()

    if combined_for_chart.empty:
        st.info("No data for this item yet.")
        return

    data_min = combined_for_chart["date"].min().date()
    data_max = combined_for_chart["date"].max().date()

    combined_filtered = combined_for_chart[
        (combined_for_chart["date"].dt.date >= start_date) & (combined_for_chart["date"].dt.date <= end_date)
    ]

    if combined_filtered.empty:
        start_date = data_min
        end_date = data_max
        combined_filtered = combined_for_chart[
            (combined_for_chart["date"].dt.date >= start_date) & (combined_for_chart["date"].dt.date <= end_date)
        ]

    today_date = pd.Timestamp.now().date()

    past_rect = alt.Chart(pd.DataFrame({"start": [pd.Timestamp(start_date)], "end": [pd.Timestamp(today_date)]})).mark_rect(
        opacity=0.1, color="blue"
    ).encode(x="start:T", x2="end:T")

    future_rect = alt.Chart(pd.DataFrame({"start": [pd.Timestamp(today_date)], "end": [pd.Timestamp(end_date)]})).mark_rect(
        opacity=0.08, color="orange"
    ).encode(x="start:T", x2="end:T")

    series_domain = sorted([s for s in combined_filtered["series"].dropna().unique().tolist()])
    series_sel = alt.selection_point(
        fields=["series"],
        bind="legend",
        toggle="true",
        value=[{"series": s} for s in series_domain],
        empty="none",
    )

    chart = (
        alt.Chart(combined_filtered)
        .mark_line(point=True)
        .transform_filter(series_sel)
        .encode(
            x=alt.X("date:T"),
            y=alt.Y("value:Q"),
            color=alt.Color("series:N", scale=alt.Scale(domain=series_domain)),
            tooltip=["series", "date", "value"],
        )
        .properties(height=400)
        .add_params(series_sel)
    )

    layered_chart = (past_rect + future_rect + chart).properties(width=700)
    st.altair_chart(layered_chart, use_container_width=True)

    st.subheader("Metrics for selected item / methods")
    if metrics_df.empty:
        st.info("No metrics yet.")
    else:
        pivot = metrics_df.pivot(index="forecast_method", columns="metric_name", values="metric_value")
        pivot = pivot.sort_index()
        if selected_methods:
            pivot = pivot.reindex(selected_methods)
        st.dataframe(pivot.style.format("{:.2f}"))
        with st.expander("Raw metric rows"):
            st.dataframe(metrics_df)

    with st.expander("Show raw data"):
        st.write("**Sales history**")
        if not history.empty:
            st.dataframe(history.sort_values("date"))
        st.write("**Forecasts**")
        if not forecasts.empty:
            st.dataframe(forecasts.sort_values("date"))


if __name__ == "__main__":
    main()
