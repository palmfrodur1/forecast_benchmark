# app.py
import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
from datetime import datetime, timedelta
from db import get_connection, init_db
from ingest import import_sales_history, import_forecasts
from metrics import recompute_all_metrics      

# Ensure tables exist
init_db()

@st.cache_data
def get_projects():
    con = get_connection()
    df = con.execute("SELECT DISTINCT project FROM sales_history ORDER BY project").fetchdf()
    con.close()
    return df["project"].tolist()

@st.cache_data
def get_items(project: str):
    con = get_connection()
    df = con.execute("""
        SELECT DISTINCT item_id
        FROM sales_history
        WHERE project = ?
        ORDER BY item_id
    """, [project]).fetchdf()
    con.close()
    return df["item_id"].tolist()

@st.cache_data
def get_forecast_methods(project: str, item_id: str):
    con = get_connection()
    df = con.execute("""
        SELECT DISTINCT forecast_method
        FROM forecasts
        WHERE project = ? AND item_id = ?
        ORDER BY forecast_method
    """, [project, item_id]).fetchdf()
    con.close()
    return df["forecast_method"].tolist()

@st.cache_data
def load_series(project: str, item_id: str, methods: list[str]):
    con = get_connection()

    history = con.execute("""
        SELECT
            sale_date AS date,
            sales    AS value,
            'History'::VARCHAR AS series
        FROM sales_history
        WHERE project = ? AND item_id = ?
        ORDER BY sale_date
    """, [project, item_id]).fetchdf()

    if methods:
        methods_tuple = tuple(methods)
        placeholders = ",".join(["?"] * len(methods))
        sql = f"""
            SELECT
                forecast_date AS date,
                forecast      AS value,
                forecast_method AS series
            FROM forecasts
            WHERE project = ?
              AND item_id = ?
              AND forecast_method IN ({placeholders})
            ORDER BY forecast_date
        """

        params = [project, item_id, *methods]
        forecasts = con.execute(sql, params).fetchdf()
    else:
        forecasts = pd.DataFrame(columns=["date", "value", "series"])

    con.close()

    combined = pd.concat([history, forecasts], ignore_index=True)
    return history, forecasts, combined

@st.cache_data
def load_metrics(project: str, item_id: str, methods: list[str]):
    if not methods:
        return pd.DataFrame(columns=["forecast_method", "metric_name", "metric_value", "n_points"])

    con = get_connection()
    placeholders = ",".join(["?"] * len(methods))
    sql = f"""
        SELECT forecast_method, metric_name, metric_value, n_points
        FROM forecast_metrics
        WHERE project = ?
          AND item_id = ?
          AND forecast_method IN ({placeholders})
        ORDER BY forecast_method, metric_name
    """
    params = [project, item_id, *methods]
    df = con.execute(sql, params).fetchdf()
    con.close()
    return df


def main():
    st.title("Forecast Benchmark Explorer")

    # Reload data button
    if st.sidebar.button("🔄 Reload Data from CSV"):
        DATA_DIR = Path(__file__).parent / "Data"
        sales_csv = DATA_DIR / "sales_history.csv"
        forecasts_csv = DATA_DIR / "forecasts.csv"
        
        with st.spinner("Reloading data..."):
            try:
                import_sales_history(sales_csv)
                import_forecasts(forecasts_csv)
                st.cache_data.clear()
                st.success("Data reloaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error reloading data: {e}")
                return

    if st.sidebar.button("Recompute metrics (all items)"):
        with st.spinner("Recomputing metrics..."):
            try:
                recompute_all_metrics()
                st.cache_data.clear()  # clear cached queries
                st.success("Metrics recomputed.")
                st.rerun()
            except Exception as e:
                st.error(f"Error recomputing metrics: {e}")
                return
      

    st.sidebar.markdown("---")

    projects = get_projects()
    if not projects:
        st.warning("No data found. Import CSVs first using `python ingest.py`.")
        return

    project = st.sidebar.selectbox("Project", projects)

    items = get_items(project)
    if not items:
        st.warning("No items for this project.")
        return

    item_id = st.sidebar.selectbox("Item", items)

    methods = get_forecast_methods(project, item_id)
    selected_methods = st.sidebar.multiselect(
        "Forecast methods",
        options=methods,
        default=methods  # select all by default
    )

    history, forecasts, combined = load_series(project, item_id, selected_methods)
    metrics_df = load_metrics(project, item_id, selected_methods)

    st.subheader(f"Item: {item_id} — Project: {project}")

    # Date range controls (by month) - placed near chart
    st.markdown("---")
    
    col1, col2, col3 = st.columns([0.08, 0.76, 0.16])
    
    with col1:
        months_back = st.number_input(
            "Months back",
            min_value=0,
            max_value=120,
            value=24,
            step=1,
            key="months_back"
        )
    
    with col3:
        months_forward = st.number_input(
            "Months forward",
            min_value=0,
            max_value=120,
            value=12,
            step=1,
            key="months_forward"
        )
    
    # Calculate actual dates from month inputs
    today = datetime.now()
    first_of_current_month = datetime(today.year, today.month, 1)
    start_date = (first_of_current_month - timedelta(days=30*months_back)).date()
    end_date = (first_of_current_month + timedelta(days=30*months_forward + 30)).date()  # Add buffer for end of period

    # Filter data by date range
    combined['date'] = pd.to_datetime(combined['date'])
    combined_filtered = combined[(combined['date'].dt.date >= start_date) & (combined['date'].dt.date <= end_date)]
    
    history_filtered = history[(pd.to_datetime(history['date']).dt.date >= start_date) & (pd.to_datetime(history['date']).dt.date <= end_date)]
    forecasts_filtered = forecasts[(pd.to_datetime(forecasts['date']).dt.date >= start_date) & (pd.to_datetime(forecasts['date']).dt.date <= end_date)]

    if combined_filtered.empty:
        st.info("No data for this selection yet.")
        return

    # Create background shading for past vs future
    today_date = pd.Timestamp.now().date()
    
    # Background rectangles for past/future
    past_rect = alt.Chart(
        pd.DataFrame({'start': [pd.Timestamp(start_date)], 'end': [pd.Timestamp(today_date)]})
    ).mark_rect(opacity=0.1, color='blue').encode(
        x='start:T',
        x2='end:T'
    )
    
    future_rect = alt.Chart(
        pd.DataFrame({'start': [pd.Timestamp(today_date)], 'end': [pd.Timestamp(end_date)]})
    ).mark_rect(opacity=0.08, color='orange').encode(
        x='start:T',
        x2='end:T'
    )
    
    # Main chart
    chart = (
        alt.Chart(combined_filtered)
        .mark_line(point=True)
        .encode(
            x="date:T",
            y="value:Q",
            color="series:N",
            tooltip=["series", "date", "value"]
        )
        .properties(height=400)
        .interactive()
    )
    
    # Layer background and chart
    layered_chart = (past_rect + future_rect + chart).properties(width=700)
    st.altair_chart(layered_chart, use_container_width=True)

    # ---- Metrics table ----
    st.subheader("Metrics for selected item / methods")

    if metrics_df.empty:
        st.info("No metrics yet. Run `python metrics.py` or use the sidebar button.")
    else:
        pivot = metrics_df.pivot(
            index="forecast_method",
            columns="metric_name",
            values="metric_value"
        )
        pivot = pivot.sort_index()
        st.dataframe(pivot.style.format("{:.2f}"))
        with st.expander("Raw metric rows"):
            st.dataframe(metrics_df)

    # Show data
    with st.expander("Show raw data"):
        st.write("**Sales history**")
        st.dataframe(history_filtered.sort_values("date"))
        st.write("**Forecasts**")
        st.dataframe(forecasts_filtered.sort_values("date"))

if __name__ == "__main__":
    main()
