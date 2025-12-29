# app.py
import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
from datetime import datetime, timedelta, date
from db import get_connection, init_db
from ingest import import_sales_history, import_forecasts
import requests
import json
import os
import time
from urllib.parse import urlparse, urlunparse


DEFAULT_NOSTRADAMUS_API_BASE_URL = os.getenv('NOSTRADAMUS_API_BASE_URL', 'https://api.nostradamus-api.com')


def _run_suffix() -> str:
    # Suffix used to make each run distinct in the DB/UI without schema changes.
    return datetime.now().strftime('%Y%m%d_%H%M%S')


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


def _aggregate_monthly_sales(df_history: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily history to monthly totals.

    - Sums `sales` per (item_id, month)
    - Sets `sale_date` to the 1st of the month (month start)
    - Returns a dataframe with columns: item_id, sale_date, sales
    """
    if df_history.empty:
        return df_history

    df = df_history.copy()
    if 'sale_date' not in df.columns or 'sales' not in df.columns or 'item_id' not in df.columns:
        return df_history

    df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
    df = df[df['sale_date'].notna()].copy()
    if df.empty:
        return df

    # Month start timestamp (YYYY-MM-01)
    # Note: Pandas uses 'M' for monthly periods; 'MS' is a timestamp freq, not a period freq.
    df['sale_date'] = df['sale_date'].dt.to_period('M').dt.to_timestamp(how='start')

    df = (
        df.groupby(['item_id', 'sale_date'], as_index=False)['sales']
        .sum()
        .sort_values(['item_id', 'sale_date'])
    )
    return df


def _aggregate_monthly_history_series(history_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate chart history series to monthly totals (month start dates)."""
    if history_df is None or history_df.empty:
        return pd.DataFrame(columns=["date", "value", "series"])

    df = history_df.copy()
    if 'date' not in df.columns or 'value' not in df.columns:
        return pd.DataFrame(columns=["date", "value", "series"])

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[df['date'].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "value", "series"])

    df['date'] = df['date'].dt.to_period('M').dt.to_timestamp(how='start')
    df = df.groupby(['date'], as_index=False)['value'].sum()
    df['series'] = 'History (Monthly)'
    return df


def _normalize_localhost_url(url: str) -> str:
    """Downgrade https->http for localhost URLs (common in Docker dev).

    If you actually run TLS locally, keep using https and remove this.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme == 'https' and (parsed.hostname in {'localhost', '127.0.0.1', '0.0.0.0'}):
            return urlunparse(parsed._replace(scheme='http'))
    except Exception:
        return url
    return url

def _call_forecast_api(
    payload: dict,
    url: str = 'https://api.nostradamus-api.com/api/v1/forecast/generate_async',
    timeout: int = 300,
    api_key: str | None = None,
) -> dict:
    """Call the Nostradamus forecasting API and return parsed JSON.

    Default URL uses the async endpoint and a longer timeout to avoid origin timeouts.
    """
    headers = {'Content-Type': 'application/json'}
    # include API key header if provided or API_KEY env var is set
    api_key = api_key or os.getenv('NOSTRADAMUS_API_KEY') or os.getenv('API_KEY')
    if api_key:
        headers['X-API-Key'] = api_key
    url = _normalize_localhost_url(url)
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _submit_job(
    payload: dict,
    base_url: str = 'https://api.nostradamus-api.com',
    webhook_url: str | None = None,
    timeout: int = 30,
    api_key: str | None = None,
) -> dict:
    """Submit a job to the async generate_job endpoint.

    Returns the JSON response which should include `job_id` and `status_url`.
    If `API_KEY` env var is set it will be sent as `X-API-Key` header.
    """
    base_url = _normalize_localhost_url(base_url)
    # build URL
    url = f"{base_url.rstrip('/')}/api/v1/forecast/generate_job"
    params = {}
    if webhook_url:
        params['webhook_url'] = webhook_url

    headers = {'Content-Type': 'application/json'}
    api_key = api_key or os.getenv('NOSTRADAMUS_API_KEY') or os.getenv('API_KEY')
    if api_key:
        headers['X-API-Key'] = api_key

    resp = requests.post(url, headers=headers, params=params, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _poll_job_status(status_url: str, timeout: int = 600, poll_initial: float = 1.0, api_key: str | None = None) -> dict:
    """Poll the job status endpoint until finished or failed, return final job hash.

    Uses `X-API-Key` header if `API_KEY` env var set. Raises on timeout.
    """
    headers = {}
    api_key = api_key or os.getenv('NOSTRADAMUS_API_KEY') or os.getenv('API_KEY')
    if api_key:
        headers['X-API-Key'] = api_key

    status_url = _normalize_localhost_url(status_url)
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


def _parse_nostradamus_response(
    resp: dict,
    project: str,
    fm_override: str | None = None,
    *,
    as_of_date: date | None = None,
    freq: str | None = None,
) -> pd.DataFrame:
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

        use_generated_dates = as_of_date is not None and bool(freq)

        # Case: lists of values (optionally with dates)
        if isinstance(vals, list):
            if use_generated_dates:
                gen_dates = _generate_forecast_dates(as_of_date, len(vals), str(freq))
                for d, v in zip(gen_dates, vals):
                    try:
                        rows.append({
                            'project': project,
                            'forecast_method': fm,
                            'item_id': str(item_id),
                            'forecast_date': d,
                            'forecast': float(v)
                        })
                    except Exception:
                        continue
                continue
            if isinstance(dates, list) and len(vals) == len(dates):
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
                continue

        # Case: list of dict points
        elif isinstance(vals, list) and all(isinstance(p, dict) for p in vals):
            if use_generated_dates:
                gen_dates = _generate_forecast_dates(as_of_date, len(vals), str(freq))
                for d, p in zip(gen_dates, vals):
                    v = p.get('value') or p.get('yhat') or p.get('forecast') or p.get('y')
                    if v is None:
                        continue
                    try:
                        rows.append({
                            'project': project,
                            'forecast_method': fm,
                            'item_id': str(item_id),
                            'forecast_date': d,
                            'forecast': float(v)
                        })
                    except Exception:
                        continue
                continue
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
                SELECT DISTINCT split_part(forecast_method, '@', 1) AS forecast_method
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
        placeholders = ",".join(["?"] * len(methods))
        sql = f"""
            WITH chosen AS (
                SELECT
                    split_part(forecast_method, '@', 1) AS base_method,
                    COALESCE(
                        MAX(CASE WHEN strpos(forecast_method, '@') = 0 THEN forecast_method END),
                        MAX(forecast_method)
                    ) AS chosen_method
                FROM forecasts
                WHERE project = ?
                  AND item_id = ?
                  AND split_part(forecast_method, '@', 1) IN ({placeholders})
                GROUP BY 1
            )
            SELECT
                f.forecast_date AS date,
                f.forecast      AS value,
                c.base_method   AS series
            FROM forecasts f
            JOIN chosen c
              ON f.forecast_method = c.chosen_method
            WHERE f.project = ?
              AND f.item_id = ?
            ORDER BY f.forecast_date
        """

        params = [project, item_id, *methods, project, item_id]
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
        WITH chosen AS (
            SELECT
                split_part(forecast_method, '@', 1) AS base_method,
                COALESCE(
                    MAX(CASE WHEN strpos(forecast_method, '@') = 0 THEN forecast_method END),
                    MAX(forecast_method)
                ) AS chosen_method
            FROM forecast_metrics
            WHERE project = ?
              AND item_id = ?
              AND split_part(forecast_method, '@', 1) IN ({placeholders})
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
        WHERE m.project = ?
          AND m.item_id = ?
        ORDER BY c.base_method, m.metric_name
    """
    params = [project, item_id, *methods, project, item_id]
    df = con.execute(sql, params).fetchdf()
    con.close()
    return df


@st.cache_data
def get_history_date_range(project: str, item_id: str | None = None):
    con = get_connection()
    if item_id is None:
        df = con.execute(
            "SELECT MIN(sale_date) AS min_date, MAX(sale_date) AS max_date FROM sales_history WHERE project = ?",
            [project],
        ).fetchdf()
    else:
        df = con.execute(
            "SELECT MIN(sale_date) AS min_date, MAX(sale_date) AS max_date FROM sales_history WHERE project = ? AND item_id = ?",
            [project, item_id],
        ).fetchdf()
    con.close()
    if df.empty or df.iloc[0].isna().any():
        return None, None
    return pd.to_datetime(df.iloc[0]['min_date']).date(), pd.to_datetime(df.iloc[0]['max_date']).date()


def _apply_as_of_cutoff(df_hist: pd.DataFrame, as_of_date):
    if df_hist is None or df_hist.empty:
        return df_hist, None
    if as_of_date is None:
        return df_hist, None
    df = df_hist.copy()
    df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
    df = df[df['sale_date'].notna()].copy()
    if df.empty:
        return df, None
    df = df[df['sale_date'].dt.date <= as_of_date].copy()
    df['sale_date'] = df['sale_date'].dt.date
    return df, as_of_date


def _effective_as_of_for_monthly(as_of_date):
    """For monthly aggregation, avoid including partial months.

    If as_of_date isn't the last day of its month, use the previous month-end.
    """
    if as_of_date is None:
        return None
    ts = pd.Timestamp(as_of_date)
    month_end = (ts + pd.offsets.MonthEnd(0)).date()
    if as_of_date == month_end:
        return as_of_date
    return (ts - pd.offsets.MonthEnd(1)).date()


def _forecast_start_after_as_of(as_of: date, freq: str) -> pd.Timestamp:
    """First forecast timestamp strictly after as_of, aligned to freq.

    Semantics:
    - Daily: next day
    - Monthly: next month start
    """
    ts = pd.Timestamp(as_of)
    f = str(freq or '').upper()
    if f.startswith('M'):
        return (ts + pd.offsets.MonthBegin(1)).normalize()
    return (ts + pd.Timedelta(days=1)).normalize()


def _generate_forecast_dates(as_of: date, periods: int, freq: str) -> list[date]:
    """Generate forecast dates from as_of + freq.

    Note: even if payload freq is 'M', we store/plot month-start dates.
    """
    start = _forecast_start_after_as_of(as_of, freq)
    f = str(freq or '').upper()
    if f.startswith('M'):
        rng = pd.date_range(start=start, periods=int(periods), freq='MS')
    else:
        rng = pd.date_range(start=start, periods=int(periods), freq=freq or 'D')
    return [pd.Timestamp(d).date() for d in rng]


def main():
    st.title("Forecast Benchmark Explorer")

    st.sidebar.markdown("---")

    # --- Global settings (used by both project + item forecast runs) ---
    if 'nostradamus_api_base_url' not in st.session_state:
        st.session_state['nostradamus_api_base_url'] = DEFAULT_NOSTRADAMUS_API_BASE_URL
    if 'nostradamus_api_key' not in st.session_state:
        st.session_state['nostradamus_api_key'] = ''
    if 'timegpt_api_key' not in st.session_state:
        st.session_state['timegpt_api_key'] = ''

    with st.sidebar.expander('Settings', expanded=True):
        st.text_input(
            'Nostradamus API base URL',
            key='nostradamus_api_base_url',
            help='Base URL for forecast jobs, e.g. https://api.nostradamus-api.com',
        )
        st.text_input(
            'Nostradamus API key (optional)',
            type='password',
            key='nostradamus_api_key',
            help='Sent as X-API-Key header to the Nostradamus API (leave blank if not required).',
        )
        st.text_input(
            'TimeGPT API key (optional)',
            type='password',
            key='timegpt_api_key',
            help='Only used when Mode=timegpt; sent in the request payload.',
        )

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

    st.sidebar.markdown("---")

    # --- Project forecast section (button + parameters directly underneath) ---
    if st.sidebar.button("Generate forecasts for project"):
        next_open = not bool(st.session_state.get('_show_project_forecast', False))
        st.session_state['_show_project_forecast'] = next_open
        if next_open:
            st.session_state['_show_item_forecast'] = False

    if st.session_state.get('_show_project_forecast'):
        with st.sidebar.container():
            st.markdown("**Project forecast parameters**")
            with st.form("forecast_form"):
                run_mode = st.selectbox("Mode", options=["local", "timegpt"], index=0)
                local_model = st.selectbox(
                    "Local model",
                    options=[
                        'auto_arima','auto_ets','naive','seasonal_naive','croston_optimized','adida','theta','optimized_theta','auto_ces'
                    ],
                    index=0,
                )
                freq = st.selectbox("Frequency", options=['D','M','W','H','Q','Y'], index=1)
                aggregate_monthly = st.checkbox(
                    "Aggregate to monthly totals",
                    value=False,
                    help="If enabled, daily sales are summed per month and sent with dates set to the 1st of the month. Frequency will be sent as M.",
                )
                season_length = st.number_input("Season length", min_value=1, max_value=365, value=12)
                forecast_periods = st.number_input("Forecast periods", min_value=1, max_value=365, value=12)
                quantiles_text = st.text_input("Quantiles (comma-separated, for TimeGPT)", value="")
                webhook_url = st.text_input("Webhook URL (optional)")
                wait_for_completion = st.checkbox("Wait for completion (poll status)", value=True)
                poll_timeout_seconds = st.number_input(
                    "Polling timeout (s)",
                    min_value=60,
                    max_value=24 * 60 * 60,
                    value=1800,
                )
                run_in_batches = st.checkbox("Run in batches", value=True)
                batch_size_items = st.number_input("Batch size (items per job)", min_value=1, max_value=5000, value=500)
                min_d, max_d = get_history_date_range(project)
                as_of_date_project = st.date_input(
                    "Forecast as-of date",
                    value=max_d if max_d else datetime.now().date(),
                    min_value=min_d,
                    max_value=max_d,
                    help="Only history up to this date is used; forecast starts the day after (freq-aligned).",
                )
                item_selection = st.multiselect(
                    "Items to include (leave empty = all items in project)",
                    options=items,
                    default=[],
                )
                submitted = st.form_submit_button("Batch run")

            if submitted:
                st.session_state['_show_project_forecast'] = False
                st.session_state['_project_forecast_request'] = {
                    'project': project,
                    'run_mode': run_mode,
                    'local_model': local_model,
                    'freq': freq,
                    'aggregate_monthly': bool(aggregate_monthly),
                    'season_length': int(season_length),
                    'forecast_periods': int(forecast_periods),
                    'quantiles_text': quantiles_text,
                    'api_key': st.session_state.get('timegpt_api_key', ''),
                    'api_base_url': st.session_state.get('nostradamus_api_base_url', DEFAULT_NOSTRADAMUS_API_BASE_URL),
                    'nostradamus_api_key': st.session_state.get('nostradamus_api_key', ''),
                    'webhook_url': webhook_url,
                    'wait_for_completion': bool(wait_for_completion),
                    'poll_timeout_seconds': int(poll_timeout_seconds),
                    'run_in_batches': bool(run_in_batches),
                    'batch_size_items': int(batch_size_items),
                    'as_of_date': as_of_date_project.isoformat() if as_of_date_project else None,
                    'item_selection': item_selection,
                }
                st.rerun()

    # --- Item forecast section (button + parameters directly underneath) ---
    if st.sidebar.button("Generate forecast for this item"):
        next_open = not bool(st.session_state.get('_show_item_forecast', False))
        st.session_state['_show_item_forecast'] = next_open
        if next_open:
            st.session_state['_show_project_forecast'] = False

    if st.session_state.get('_show_item_forecast'):
        with st.sidebar.container():
            st.markdown("**Item forecast parameters**")
            with st.form("forecast_item_form"):
                run_mode_item = st.selectbox("Mode", options=["local", "timegpt"], index=0)
                local_model_item = st.selectbox(
                    "Local model",
                    options=[
                        'auto_arima','auto_ets','naive','seasonal_naive','croston_optimized','adida','theta','optimized_theta','auto_ces'
                    ],
                    index=0,
                )
                freq_item = st.selectbox("Frequency", options=['D','M','W','H','Q','Y'], index=1)
                aggregate_monthly_item = st.checkbox(
                    "Aggregate to monthly totals",
                    value=False,
                    help="If enabled, daily sales are summed per month and sent with dates set to the 1st of the month. Frequency will be sent as M.",
                )
                season_length_item = st.number_input("Season length", min_value=1, max_value=365, value=12)
                forecast_periods_item = st.number_input("Forecast periods", min_value=1, max_value=365, value=12)
                min_di, max_di = get_history_date_range(project, item_id=item_id)
                as_of_date_item = st.date_input(
                    "Forecast as-of date",
                    value=max_di if max_di else datetime.now().date(),
                    min_value=min_di,
                    max_value=max_di,
                    help="Only history up to this date is used; forecast starts the day after (freq-aligned).",
                )
                quantiles_text_item = st.text_input("Quantiles (comma-separated, for TimeGPT)", value="")
                timeout_seconds_item = st.number_input("Request timeout (s)", min_value=30, max_value=1200, value=300)
                submitted_item = st.form_submit_button("Single run")

            if submitted_item:
                st.session_state['_show_item_forecast'] = False
                st.session_state['_item_forecast_request'] = {
                    'project': project,
                    'item_id': item_id,
                    'run_mode': run_mode_item,
                    'local_model': local_model_item,
                    'freq': freq_item,
                    'aggregate_monthly': bool(aggregate_monthly_item),
                    'season_length': int(season_length_item),
                    'forecast_periods': int(forecast_periods_item),
                    'as_of_date': as_of_date_item.isoformat() if as_of_date_item else None,
                    'quantiles_text': quantiles_text_item,
                    'api_key': st.session_state.get('timegpt_api_key', ''),
                    'api_base_url': st.session_state.get('nostradamus_api_base_url', DEFAULT_NOSTRADAMUS_API_BASE_URL),
                    'nostradamus_api_key': st.session_state.get('nostradamus_api_key', ''),
                    'timeout_seconds': int(timeout_seconds_item),
                }
                st.rerun()

    # --- Forecast execution (main area) ---
    project_req = st.session_state.pop('_project_forecast_request', None)
    if project_req:
        st.markdown('---')
        st.subheader('Project forecast run')

        project_run_tag = _run_suffix()
        if project_req.get('as_of_date'):
            project_run_tag = f"{project_run_tag}_asof{project_req['as_of_date']}"
        st.caption(f"Run tag: @{project_run_tag} (used to keep this run distinct)")

        api_base_url = _normalize_localhost_url(project_req['api_base_url'])
        if api_base_url != project_req['api_base_url']:
            st.warning(f"Using {api_base_url} (localhost usually doesn't use TLS).")

        # Load history once (then split into item batches).
        con = get_connection()
        if project_req['item_selection']:
            placeholders = ','.join(['?'] * len(project_req['item_selection']))
            sql = f"SELECT item_id, sale_date, sales FROM sales_history WHERE project = ? AND item_id IN ({placeholders}) ORDER BY item_id, sale_date"
            params = [project_req['project'], *project_req['item_selection']]
        else:
            sql = "SELECT item_id, sale_date, sales FROM sales_history WHERE project = ? ORDER BY item_id, sale_date"
            params = [project_req['project']]
        df_hist = con.execute(sql, params).fetchdf()
        con.close()

        if df_hist.empty:
            st.error("No sales history found for the selected project/items.")
        else:
            as_of_date = None
            if project_req.get('as_of_date'):
                try:
                    as_of_date = pd.to_datetime(project_req['as_of_date']).date()
                except Exception:
                    as_of_date = None

            if as_of_date is not None:
                df_hist, _ = _apply_as_of_cutoff(df_hist, as_of_date)
                if df_hist.empty:
                    st.error("No history rows left after applying as-of date.")
                    return

            if project_req['aggregate_monthly']:
                df_hist = _aggregate_monthly_sales(df_hist)

            item_ids = df_hist['item_id'].dropna().astype(str).unique().tolist()
            if not item_ids:
                st.error("No valid item_id values found in sales history.")
            else:
                if project_req['run_in_batches']:
                    n = int(project_req['batch_size_items'])
                    item_batches = [item_ids[i:i + n] for i in range(0, len(item_ids), n)]
                else:
                    item_batches = [item_ids]

                st.info(
                    f"Submitting {len(item_batches)} job(s) for {len(item_ids)} item(s). "
                    f"(Batching={'on' if project_req['run_in_batches'] else 'off'}, items/job={int(project_req['batch_size_items'])})"
                )

                progress = st.progress(0)
                any_inserted = 0

                for idx, batch_items in enumerate(item_batches, start=1):
                    df_hist_batch = df_hist[df_hist['item_id'].astype(str).isin(set(map(str, batch_items)))].copy()
                    sim_input_his = _format_sim_input(df_hist_batch)
                    if len(sim_input_his) < 5:
                        st.warning(f"Batch {idx}: less than 5 data points — some models may not work well.")

                    payload = {
                        'sim_input_his': sim_input_his,
                        'forecast_periods': int(project_req['forecast_periods']),
                        'mode': project_req['run_mode'],
                        'local_model': project_req['local_model'],
                        'season_length': int(project_req['season_length']),
                        'freq': 'M' if project_req['aggregate_monthly'] else project_req['freq'],
                    }
                    if project_req['quantiles_text'].strip():
                        try:
                            quantiles = [float(q.strip()) for q in project_req['quantiles_text'].split(',') if q.strip()]
                            payload['quantiles'] = quantiles
                        except Exception:
                            st.error("Invalid quantiles format. Use comma-separated floats like: 0.1,0.5,0.9")
                            break

                    if project_req['run_mode'] == 'timegpt' and project_req['api_key']:
                        payload['api_key'] = project_req['api_key']

                    st.write(f"Batch {idx}/{len(item_batches)}: submitting job for {len(batch_items)} item(s)...")
                    try:
                        resp = _submit_job(
                            payload,
                            base_url=api_base_url,
                            webhook_url=project_req['webhook_url'],
                            api_key=(project_req.get('nostradamus_api_key') or None),
                        )
                    except Exception as e:
                        st.error(f"Job submission failed (batch {idx}): {e}")
                        break

                    status_url = resp.get('status_url') or resp.get('status') or resp.get('status_endpoint')
                    job_id = resp.get('job_id')
                    if not status_url:
                        st.warning(f"Batch {idx}: no status_url returned. job_id={job_id}")
                        break

                    if project_req['wait_for_completion']:
                        try:
                            job = _poll_job_status(
                                status_url,
                                timeout=int(project_req['poll_timeout_seconds']),
                                api_key=(project_req.get('nostradamus_api_key') or None),
                            )
                        except Exception as e:
                            st.error(f"Polling failed (batch {idx}): {e}")
                            break

                        if job.get('status') != 'finished' or not job.get('result'):
                            st.error(f"Batch {idx}: job ended with status={job.get('status')} error={job.get('error')}")
                            break

                        df_new = _parse_nostradamus_response(
                            job.get('result'),
                            project=project_req['project'],
                            fm_override=project_req['local_model'] if project_req['run_mode'] == 'local' else None,
                            as_of_date=as_of_date,
                            freq=payload.get('freq'),
                        )
                        if df_new.empty:
                            st.warning(f"Batch {idx}: parsed 0 forecast rows.")
                        else:
                            con = get_connection()
                            try:
                                df_new = df_new.copy()
                                con.register('forecasts_api_df_batch', df_new)
                                # Overwrite existing forecasts for these (item, method) pairs.
                                # Delete both untagged + any previously tagged runs for the same base method.
                                con.execute(
                                    """
                                    DELETE FROM forecasts
                                    WHERE project = ?
                                      AND (item_id, split_part(forecast_method, '@', 1)) IN (
                                          SELECT DISTINCT item_id, forecast_method FROM forecasts_api_df_batch
                                      )
                                    """,
                                    [project_req['project']],
                                )
                                con.execute(
                                    'INSERT INTO forecasts (project, forecast_method, item_id, forecast_date, forecast) '
                                    'SELECT project, forecast_method, item_id, forecast_date, forecast FROM forecasts_api_df_batch'
                                )
                            finally:
                                con.close()

                            any_inserted += len(df_new)
                            st.success(f"Batch {idx}: inserted {len(df_new)} rows (overwrote existing for same method).")
                    else:
                        st.success(f"Batch {idx}: submitted job_id={job_id}. (Not waiting for completion)")

                    progress.progress(int(idx / len(item_batches) * 100))

                if any_inserted > 0:
                    st.success(f"Inserted {any_inserted} forecast rows total.")
                    st.cache_data.clear()
                    st.rerun()

    history, forecasts, combined = load_series(project, item_id, selected_methods)
    metrics_df = load_metrics(project, item_id, selected_methods)

    st.subheader(f"Item: {item_id} — Project: {project}")

    item_req = st.session_state.pop('_item_forecast_request', None)
    if item_req:
        st.markdown('---')
        st.subheader('Item forecast run')

        con = get_connection()
        sql = "SELECT item_id, sale_date, sales FROM sales_history WHERE project = ? AND item_id = ? ORDER BY sale_date"
        params = [item_req['project'], item_req['item_id']]
        df_hist_item = con.execute(sql, params).fetchdf()
        con.close()

        if df_hist_item.empty:
            st.error("No sales history found for this item.")
        else:
            as_of_date_item = None
            if item_req.get('as_of_date'):
                try:
                    as_of_date_item = pd.to_datetime(item_req['as_of_date']).date()
                except Exception:
                    as_of_date_item = None

            if as_of_date_item is not None:
                df_hist_item, _ = _apply_as_of_cutoff(df_hist_item, as_of_date_item)
                if df_hist_item.empty:
                    st.error("No history rows left after applying as-of date.")
                    return

            if item_req['aggregate_monthly']:
                df_hist_item = _aggregate_monthly_sales(df_hist_item)

            sim_input_his = _format_sim_input(df_hist_item)
            if len(sim_input_his) < 1:
                st.error("No valid sale_date rows to send. Check your data.")
            else:
                payload = {
                    'sim_input_his': sim_input_his,
                    'forecast_periods': int(item_req['forecast_periods']),
                    'mode': item_req['run_mode'],
                    'local_model': item_req['local_model'],
                    'season_length': int(item_req['season_length']),
                    'freq': 'M' if item_req['aggregate_monthly'] else item_req['freq'],
                }
                if item_req['quantiles_text'].strip():
                    try:
                        quantiles = [float(q.strip()) for q in item_req['quantiles_text'].split(',') if q.strip()]
                        payload['quantiles'] = quantiles
                    except Exception:
                        st.error("Invalid quantiles format. Use comma-separated floats like: 0.1,0.5,0.9")
                        payload = None

                if payload is not None:
                    if item_req['run_mode'] == 'timegpt' and item_req['api_key']:
                        payload['api_key'] = item_req['api_key']
                    api_base_url_item = _normalize_localhost_url(item_req['api_base_url'])
                    if api_base_url_item != item_req['api_base_url']:
                        st.warning(f"Using {api_base_url_item} (localhost usually doesn't use TLS).")

                    # Diagnostics to understand "one item" job size and duration.
                    st.caption(f"Prepared sim_input_his with {len(sim_input_his)} rows.")
                    if len(sim_input_his) > 5000:
                        st.warning(
                            "This item has a lot of history rows; even a single-item forecast can take a while. "
                            "If you see timeouts, consider aggregating monthly or reducing history length on the API side."
                        )

                    # Use async job flow to avoid Cloudflare 524 timeouts on long model fits.
                    nostradamus_key = item_req.get('nostradamus_api_key') or None
                    run_start = time.perf_counter()
                    st.info("Submitting forecast job and waiting for result...")
                    try:
                        submit_start = time.perf_counter()
                        resp = _submit_job(
                            payload,
                            base_url=api_base_url_item,
                            webhook_url=None,
                            timeout=min(30, int(item_req['timeout_seconds'])),
                            api_key=nostradamus_key,
                        )
                        st.caption(
                            f"Submitted job_id={resp.get('job_id')} in {time.perf_counter() - submit_start:.2f}s"
                        )
                    except Exception as e:
                        st.error(f"Forecast job submission failed: {e}")
                        resp = None

                    status_url = (resp or {}).get('status_url')
                    if resp is not None and not status_url:
                        st.error(f"No status_url returned by API. job_id={(resp or {}).get('job_id')}")
                        resp = None

                    if resp is not None and status_url:
                        try:
                            poll_start = time.perf_counter()
                            job = _poll_job_status(
                                status_url,
                                timeout=int(item_req['timeout_seconds']),
                                api_key=nostradamus_key,
                            )
                            st.caption(
                                f"Polling finished in {time.perf_counter() - poll_start:.2f}s "
                                f"(total {time.perf_counter() - run_start:.2f}s)"
                            )
                        except Exception as e:
                            st.error(f"Forecast polling failed: {e}")
                            job = None

                        if not job or job.get('status') != 'finished' or not job.get('result'):
                            st.error(
                                f"Job ended with status={(job or {}).get('status')} error={(job or {}).get('error')}"
                            )
                            resp = None
                        else:
                            resp = job.get('result')

                    if resp:
                        with st.expander("Response", expanded=False):
                            st.json(resp)
                        df_new = _parse_nostradamus_response(
                            resp,
                            project=item_req['project'],
                            fm_override=item_req['local_model'] if item_req['run_mode'] == 'local' else None,
                            as_of_date=as_of_date_item,
                            freq=payload.get('freq'),
                        )

                        if df_new.empty:
                            st.warning("Could not parse any forecast rows from response.")
                        else:
                            con = get_connection()
                            df_new = df_new.copy()
                            con.register('forecasts_api_df_item', df_new)
                            con.execute(
                                """
                                DELETE FROM forecasts
                                WHERE project = ?
                                  AND item_id = ?
                                  AND split_part(forecast_method, '@', 1) IN (
                                      SELECT DISTINCT forecast_method FROM forecasts_api_df_item
                                  )
                                """,
                                [item_req['project'], item_req['item_id']],
                            )
                            con.execute(
                                "INSERT INTO forecasts (project, forecast_method, item_id, forecast_date, forecast) "
                                "SELECT project, forecast_method, item_id, forecast_date, forecast FROM forecasts_api_df_item"
                            )
                            con.close()
                            st.success(
                                f"Saved {len(df_new)} forecast rows (overwrote existing for same method)."
                            )
                            st.cache_data.clear()
                            st.rerun()

    # (item-level parameter form lives in the sidebar)

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

    show_monthly_history = st.checkbox(
        "Show monthly aggregated history",
        value=True,
        help="Overlays a monthly-summed history series dated to the 1st of each month.",
    )

    combined_for_chart = combined.copy()
    if show_monthly_history:
        monthly_hist = _aggregate_monthly_history_series(history)
        if not monthly_hist.empty:
            combined_for_chart = pd.concat([combined_for_chart, monthly_hist], ignore_index=True)

    # Filter data by date range
    combined_for_chart['date'] = pd.to_datetime(combined_for_chart['date'], errors='coerce')
    combined_for_chart = combined_for_chart[combined_for_chart['date'].notna()].copy()

    if combined_for_chart.empty:
        st.info("No data for this item yet.")
        return

    data_min = combined_for_chart['date'].min().date()
    data_max = combined_for_chart['date'].max().date()

    combined_filtered = combined_for_chart[
        (combined_for_chart['date'].dt.date >= start_date) & (combined_for_chart['date'].dt.date <= end_date)
    ]

    if combined_filtered.empty:
        # If the user-selected window filters out everything, fall back to the real data range
        start_date = data_min
        end_date = data_max
        combined_filtered = combined_for_chart[
            (combined_for_chart['date'].dt.date >= start_date) & (combined_for_chart['date'].dt.date <= end_date)
        ]

    history_filtered = history[(pd.to_datetime(history['date']).dt.date >= start_date) & (pd.to_datetime(history['date']).dt.date <= end_date)]
    forecasts_filtered = forecasts[(pd.to_datetime(forecasts['date']).dt.date >= start_date) & (pd.to_datetime(forecasts['date']).dt.date <= end_date)]

    if combined_filtered.empty:
        st.info("No data to plot.")
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
    # Legend click toggles each series on/off, while keeping all legend entries.
    series_domain = sorted([s for s in combined_filtered['series'].dropna().unique().tolist()])
    series_sel = alt.selection_point(
        fields=["series"],
        bind="legend",
        # Click toggles membership (no need for shift-click). Using string for compatibility.
        toggle="true",
        # Start with everything visible. Clicking a legend item toggles it off/on.
        value=[{"series": s} for s in series_domain],
        # If everything is toggled off, show nothing (but legend remains).
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
        # Ensure the table includes all selected methods, even if some have no metrics yet.
        if selected_methods:
            pivot = pivot.reindex(selected_methods)
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
