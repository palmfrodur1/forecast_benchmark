# app.py
import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
from datetime import datetime, timedelta
from db import get_connection, init_db
from ingest import import_sales_history, import_forecasts
from metrics import recompute_all_metrics      
import requests
import json
import os
import time


def _format_sim_input(df_history: pd.DataFrame) -> list:
    """Return sim_input_his list of dicts from a sales_history dataframe.

    Expects dataframe with columns: item_id, sale_date, sales
    """
    out = []
    if df_history.empty:
        return out
    # Ensure date and sorting; drop rows with invalid dates
    df = df_history.copy()
    # coerce to datetime and drop NaT
    df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
    df = df[df['sale_date'].notna()].copy()
    if df.empty:
        return out
    df['sale_date'] = df['sale_date'].dt.date
    df = df.sort_values(['item_id', 'sale_date'])
    for _, row in df.iterrows():
        out.append({
            'item_id': row['item_id'],
            'actual_sale': float(row['sales']),
            'day': row['sale_date'].isoformat()
        })
    return out


def _call_forecast_api(payload: dict, url: str = 'https://api.nostradamus-api.com/api/v1/forecast/generate_async', timeout: int = 300) -> dict:
    """Call the Nostradamus forecasting API and return parsed JSON.

    Default URL uses the async endpoint and a longer timeout to avoid origin timeouts.
    """
    headers = {'Content-Type': 'application/json'}
    # include API key header if server has API_KEY env var set
    api_key = os.getenv('API_KEY')
    if api_key:
        headers['X-API-Key'] = api_key
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _submit_job(payload: dict, base_url: str = 'https://api.nostradamus-api.com', webhook_url: str | None = None, timeout: int = 30) -> dict:
    """Submit a job to the async generate_job endpoint.

    Returns the JSON response which should include `job_id` and `status_url`.
    If `API_KEY` env var is set it will be sent as `X-API-Key` header.
    """
    # build URL
    url = f"{base_url.rstrip('/')}/api/v1/forecast/generate_job"
    params = {}
    if webhook_url:
        params['webhook_url'] = webhook_url

    headers = {'Content-Type': 'application/json'}
    api_key = os.getenv('API_KEY')
    if api_key:
        headers['X-API-Key'] = api_key

    resp = requests.post(url, headers=headers, params=params, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _poll_job_status(status_url: str, timeout: int = 600, poll_initial: float = 1.0) -> dict:
    """Poll the job status endpoint until finished or failed, return final job hash.

    Uses `X-API-Key` header if `API_KEY` env var set. Raises on timeout.
    """
    headers = {}
    api_key = os.getenv('API_KEY')
    if api_key:
        headers['X-API-Key'] = api_key

    deadline = time.time() + timeout
    interval = poll_initial
    while time.time() < deadline:
        resp = requests.get(status_url, headers=headers, timeout=30)
        resp.raise_for_status()
        job = resp.json()
        status = job.get('status')
        if status in ('finished', 'failed'):
            return job
        time.sleep(interval)
        # gradually increase interval but cap it
        interval = min(interval * 1.7, 10)

    raise TimeoutError(f"Polling job status timed out after {timeout} seconds")


def _parse_nostradamus_response(resp: dict, project: str, fm_override: str | None = None) -> pd.DataFrame:
    """Parse Nostradamus API response into a DataFrame suitable for inserting into `forecasts`.

    Handles response shapes like the example you provided:
    {
      "forecasts": [
        {"item_id":"001","forecast":[50.0,...],"forecast_dates":["2025-11-01",...],"model_used":"auto_arima",...}
      ],
      ...
    }

    Returns DataFrame with columns: project, forecast_method, item_id, forecast_date, forecast
    """
    rows = []
    if not isinstance(resp, dict):
        return pd.DataFrame(rows)

    forecasts = resp.get('forecasts') or []
    top_model = resp.get('model')

    for it in forecasts:
        item_id = it.get('item_id') or it.get('item') or it.get('sku')
        # prefer per-item model_used then top-level model then override
        fm = it.get('model_used') or top_model or fm_override or 'unknown'

        vals = it.get('forecast') or it.get('forecasts') or it.get('values')
        dates = it.get('forecast_dates') or it.get('dates') or it.get('forecast_date')

        # Case: parallel lists of values and dates
        if isinstance(vals, list) and isinstance(dates, list) and len(vals) == len(dates):
            for d, v in zip(dates, vals):
                try:
                    rows.append({
                        'project': project,
                        'forecast_method': fm,
                        'item_id': str(item_id),
                        'forecast_date': pd.to_datetime(d).date(),
                        'forecast': float(v)
                    })
                except Exception:
                    continue

        # Case: list of dict points
        elif isinstance(vals, list) and all(isinstance(p, dict) for p in vals):
            for p in vals:
                d = p.get('day') or p.get('date') or p.get('ds')
                v = p.get('value') or p.get('yhat') or p.get('forecast') or p.get('y')
                if d is not None and v is not None:
                    try:
                        rows.append({
                            'project': project,
                            'forecast_method': fm,
                            'item_id': str(item_id),
                            'forecast_date': pd.to_datetime(d).date(),
                            'forecast': float(v)
                        })
                    except Exception:
                        continue

        # Case: vals is dict mapping date->value
        elif isinstance(vals, dict):
            for d, v in vals.items():
                try:
                    rows.append({
                        'project': project,
                        'forecast_method': fm,
                        'item_id': str(item_id),
                        'forecast_date': pd.to_datetime(d).date(),
                        'forecast': float(v)
                    })
                except Exception:
                    continue

        # Fallback: if only dates list present and no 'forecast' key, try other structures
        elif isinstance(dates, list) and isinstance(vals, (int, float, str)):
            # single value repeated? unlikely — skip
            continue

    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(rows)


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

    # --- Forecast generation UI (project-level) ---
    st.markdown("---")
    gen_col1, gen_col2 = st.columns([1, 1])
    with gen_col2:
        # Use session state to emulate a modal when Streamlit doesn't support `st.modal` in this runtime
        if st.button("Generate forecasts for project"):
            st.session_state['_show_project_forecast'] = True

        if st.session_state.get('_show_project_forecast'):
            with st.expander("Generate forecasts", expanded=True):
                st.write("Set forecast parameters and run for selected project/item(s)")
                with st.form("forecast_form"):
                    run_mode = st.selectbox("Mode", options=["local", "timegpt"], index=0)
                    local_model = st.selectbox("Local model", options=[
                        'auto_arima','auto_ets','naive','seasonal_naive','croston_optimized','adida','theta','optimized_theta','auto_ces'
                    ], index=0)
                    freq = st.selectbox("Frequency", options=['D','MS','W','H','Q','Y'], index=1)
                    season_length = st.number_input("Season length", min_value=1, max_value=365, value=12)
                    forecast_periods = st.number_input("Forecast periods", min_value=1, max_value=365, value=12)
                    quantiles_text = st.text_input("Quantiles (comma-separated, for TimeGPT)", value="")
                    api_key = st.text_input("TimeGPT API key (optional)", type="password")
                    api_base_url = st.text_input("Forecast API base URL", value="https://api.nostradamus-api.com")
                    webhook_url = st.text_input("Webhook URL (optional)")
                    wait_for_completion = st.checkbox("Wait for completion (poll status)", value=True)
                    item_selection = st.multiselect("Items to include (leave empty = all items in project)", options=items, default=[])
                    submitted = st.form_submit_button("Run and preview")

                if submitted:
                    # hide the expander after submission
                    st.session_state['_show_project_forecast'] = False
                    con = get_connection()
                    if item_selection:
                        placeholders = ','.join(['?'] * len(item_selection))
                        sql = f"SELECT item_id, sale_date, sales FROM sales_history WHERE project = ? AND item_id IN ({placeholders}) ORDER BY item_id, sale_date"
                        params = [project, *item_selection]
                    else:
                        sql = "SELECT item_id, sale_date, sales FROM sales_history WHERE project = ? ORDER BY item_id, sale_date"
                        params = [project]
                    df_hist = con.execute(sql, params).fetchdf()
                    con.close()

                    if df_hist.empty:
                        st.error("No sales history found for the selected project/items.")
                    else:
                        sim_input_his = _format_sim_input(df_hist)
                        if len(sim_input_his) < 5:
                            st.warning("Less than 5 data points — some models may not work well.")

                        payload = {
                            'sim_input_his': sim_input_his,
                            'forecast_periods': int(forecast_periods),
                            'mode': run_mode,
                            'local_model': local_model,
                            'season_length': int(season_length),
                            'freq': freq,
                        }
                        if quantiles_text.strip():
                            try:
                                quantiles = [float(q.strip()) for q in quantiles_text.split(',') if q.strip()]
                                payload['quantiles'] = quantiles
                            except Exception:
                                st.error("Invalid quantiles format. Use comma-separated floats like: 0.1,0.5,0.9")

                        if run_mode == 'timegpt' and api_key:
                            payload['api_key'] = api_key

                        st.info("Submitting job to forecast API — this may take a while to complete...")
                        try:
                            resp = _submit_job(payload, base_url=api_base_url, webhook_url=webhook_url)
                        except Exception as e:
                            st.error(f"Job submission failed: {e}")
                            resp = None

                        if resp:
                            st.write("Submission result:")
                            st.json(resp)
                            status_url = resp.get('status_url') or resp.get('status') or resp.get('status_endpoint')
                            job_id = resp.get('job_id')
                            if wait_for_completion and status_url:
                                try:
                                    st.info("Polling job status until finished...")
                                    job = _poll_job_status(status_url, timeout=600)
                                except Exception as e:
                                    st.error(f"Polling failed: {e}")
                                    job = None

                                if job:
                                    st.write("Final job object:")
                                    st.json(job)
                                    if job.get('status') == 'finished' and job.get('result'):
                                        df_new = _parse_nostradamus_response(job.get('result'), project=project, fm_override=local_model if run_mode=='local' else None)
                                    elif job.get('status') == 'finished' and not job.get('result'):
                                        st.warning("Job finished but no result field present in job. Inspect job JSON above.")
                                        df_new = pd.DataFrame()
                                    else:
                                        st.error(f"Job ended with status: {job.get('status')}")
                                        df_new = pd.DataFrame()
                                else:
                                    df_new = pd.DataFrame()
                            else:
                                # not waiting — inform user about job id / status url
                                st.success(f"Job submitted (job_id={job_id}). Poll {status_url} to check status or wait for webhook delivery.")
                                df_new = pd.DataFrame()

                            if not df_new.empty:
                                st.write("Preview of forecasts to import:")
                                st.dataframe(df_new.head(200))
                                if st.button("Import forecasts into DB"):
                                    con = get_connection()
                                    fm_name = df_new['forecast_method'].iloc[0]
                                    con.execute("DELETE FROM forecasts WHERE project = ? AND forecast_method = ?;", [project, fm_name])
                                    con.register('forecasts_api_df', df_new)
                                    con.execute("INSERT INTO forecasts (project, forecast_method, item_id, forecast_date, forecast) SELECT project, forecast_method, item_id, forecast_date, forecast FROM forecasts_api_df")
                                    con.close()
                                    st.success(f"Imported {len(df_new)} forecast rows into forecasts table (method={fm_name}).")

    history, forecasts, combined = load_series(project, item_id, selected_methods)
    metrics_df = load_metrics(project, item_id, selected_methods)

    st.subheader(f"Item: {item_id} — Project: {project}")

    # Per-item forecast: open modal to send only this item's history to the async API
    if st.button("Generate forecast for this item"):
        st.session_state['_show_item_forecast'] = True

    if st.session_state.get('_show_item_forecast'):
        with st.expander("Generate forecast for item", expanded=True):
            st.write(f"Project: {project} — Item: {item_id}")
            with st.form("forecast_item_form"):
                run_mode = st.selectbox("Mode", options=["local", "timegpt"], index=0)
                local_model = st.selectbox("Local model", options=[
                    'auto_arima','auto_ets','naive','seasonal_naive','croston_optimized','adida','theta','optimized_theta','auto_ces'
                ], index=0)
                freq = st.selectbox("Frequency", options=['D','MS','W','H','Q','Y'], index=1)
                season_length = st.number_input("Season length", min_value=1, max_value=365, value=12)
                forecast_periods = st.number_input("Forecast periods", min_value=1, max_value=365, value=12)
                quantiles_text = st.text_input("Quantiles (comma-separated, for TimeGPT)", value="")
                api_key = st.text_input("TimeGPT API key (optional)", type="password")
                timeout_seconds = st.number_input("Request timeout (s)", min_value=30, max_value=1200, value=300)
                submitted_item = st.form_submit_button("Run and preview")

            if submitted_item:
                con = get_connection()
                sql = "SELECT item_id, sale_date, sales FROM sales_history WHERE project = ? AND item_id = ? ORDER BY sale_date"
                params = [project, item_id]
                df_hist_item = con.execute(sql, params).fetchdf()
                con.close()

                if df_hist_item.empty:
                    st.error("No sales history found for this item.")
                else:
                    sim_input_his = _format_sim_input(df_hist_item)
                    if len(sim_input_his) < 1:
                        st.error("No valid sale_date rows to send. Check your data.")
                    else:
                        payload = {
                            'sim_input_his': sim_input_his,
                            'forecast_periods': int(forecast_periods),
                            'mode': run_mode,
                            'local_model': local_model,
                            'season_length': int(season_length),
                            'freq': freq,
                        }
                        if quantiles_text.strip():
                            try:
                                quantiles = [float(q.strip()) for q in quantiles_text.split(',') if q.strip()]
                                payload['quantiles'] = quantiles
                            except Exception:
                                st.error("Invalid quantiles format. Use comma-separated floats like: 0.1,0.5,0.9")

                        if run_mode == 'timegpt' and api_key:
                            payload['api_key'] = api_key

                        # additional fields for async job submission
                        api_base_url = st.text_input("Forecast API base URL", value="https://api.nostradamus-api.com")
                        webhook_url = st.text_input("Webhook URL (optional)")
                        wait_for_completion = st.checkbox("Wait for completion (poll status)", value=True)

                        st.info("Submitting job to forecast API — this may take a short while...")
                        try:
                            resp = _submit_job(payload, base_url=api_base_url, webhook_url=webhook_url, timeout=timeout_seconds)
                        except Exception as e:
                            st.error(f"Job submission failed: {e}")
                            resp = None

                        if resp:
                            st.write("Submission result:")
                            st.json(resp)
                            status_url = resp.get('status_url') or resp.get('status') or resp.get('status_endpoint')
                            job_id = resp.get('job_id')
                            if wait_for_completion and status_url:
                                try:
                                    st.info("Polling job status until finished...")
                                    job = _poll_job_status(status_url, timeout=600)
                                except Exception as e:
                                    st.error(f"Polling failed: {e}")
                                    job = None

                                if job:
                                    st.write("Final job object:")
                                    st.json(job)
                                    if job.get('status') == 'finished' and job.get('result'):
                                        df_new = _parse_nostradamus_response(job.get('result'), project=project, fm_override=local_model if run_mode=='local' else None)
                                    elif job.get('status') == 'finished' and not job.get('result'):
                                        st.warning("Job finished but no result field present in job. Inspect job JSON above.")
                                        df_new = pd.DataFrame()
                                    else:
                                        st.error(f"Job ended with status: {job.get('status')}")
                                        df_new = pd.DataFrame()
                                else:
                                    df_new = pd.DataFrame()
                            else:
                                st.success(f"Job submitted (job_id={job_id}). Poll {status_url} to check status or wait for webhook delivery.")
                                df_new = pd.DataFrame()

                            if not df_new.empty:
                                st.write("Preview of forecasts to import:")
                                st.dataframe(df_new.head(200))
                                if st.button("Import forecasts into DB"):
                                    con = get_connection()
                                    fm_name = df_new['forecast_method'].iloc[0]
                                    # delete only this item for safety
                                    con.execute("DELETE FROM forecasts WHERE project = ? AND forecast_method = ? AND item_id = ?", [project, fm_name, item_id])
                                    con.register('forecasts_api_df_item', df_new)
                                    con.execute("INSERT INTO forecasts (project, forecast_method, item_id, forecast_date, forecast) SELECT project, forecast_method, item_id, forecast_date, forecast FROM forecasts_api_df_item")
                                    con.close()
                                    st.success(f"Imported {len(df_new)} forecast rows into forecasts table (method={fm_name}).")

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
