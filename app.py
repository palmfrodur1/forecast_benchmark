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
import html
from urllib.parse import urlparse, urlunparse
from client_lightgpt import call_lightgpt_batch

from nav import render_sidebar_nav


DEFAULT_NOSTRADAMUS_API_BASE_URL = os.getenv('NOSTRADAMUS_API_BASE_URL', 'https://api.nostradamus-api.com')


st.set_page_config(page_title='View per Item')

# Used by nav.render_sidebar_nav() to link to the correct entrypoint file.
st.session_state['_entrypoint'] = 'app.py'

render_sidebar_nav()


def _ensure_single_choice_state(key: str, options: list, default=None) -> None:
    if not options:
        return
    if key not in st.session_state:
        st.session_state[key] = default if (default in options) else options[0]
        return
    if st.session_state[key] not in options:
        st.session_state[key] = default if (default in options) else options[0]


def _ensure_multi_choice_state(key: str, options: list, default: list | None = None) -> None:
    if not options:
        return
    if key not in st.session_state:
        st.session_state[key] = default if default is not None else list(options)
        return
    current = st.session_state.get(key)
    if not isinstance(current, (list, tuple)):
        current = []
    filtered = [v for v in current if v in options]
    st.session_state[key] = filtered
    if not st.session_state[key]:
        st.session_state[key] = default if default is not None else list(options)


def _sync_project_widget(widget_key: str, projects: list[str]) -> None:
    if not projects:
        return
    if 'active_project' not in st.session_state or st.session_state.get('active_project') not in projects:
        st.session_state['active_project'] = projects[0]
    # Keep the page-local widget in sync with the shared value.
    if widget_key not in st.session_state or st.session_state.get(widget_key) not in projects:
        st.session_state[widget_key] = st.session_state['active_project']
    elif st.session_state.get(widget_key) != st.session_state.get('active_project'):
        st.session_state[widget_key] = st.session_state['active_project']


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
    # Defensive: if upstream code accidentally produced duplicate column labels
    # (e.g., two 'item_id' columns), Pandas operations like sort_values will fail.
    if not df.columns.is_unique:
        df = df.loc[:, ~df.columns.duplicated()].copy()
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
        model_used = it.get('model_used')

        # Special case: if the user requested AutoModel, keep that provenance
        # visible but record the actual model returned by Nostradamus.
        # Example: user selects auto_model, API returns model_used=auto_arima -> AM:auto_arima
        if str(fm_override or '').strip().lower() == 'auto_model':
            resolved = model_used or top_model or 'unknown'
            fm = f"AM:{resolved}"
        else:
            # Prefer per-item model_used if present; otherwise prefer explicit override
            # (the user-selected local model) before falling back to a top-level model.
            fm = model_used or fm_override or top_model or 'unknown'

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


@st.cache_data(ttl=60)
def get_project_forecast_methods(project: str) -> list[str]:
    """Return all base forecast methods present in forecasts for a project.

    Used to keep series colors stable across different items.
    """
    con = get_connection()
    df = con.execute(
        """
        SELECT DISTINCT split_part(forecast_method, '@', 1) AS forecast_method
        FROM forecasts
        WHERE project = ?
        ORDER BY forecast_method
        """,
        [project],
    ).fetchdf()
    con.close()
    if df.empty:
        return []
    return df["forecast_method"].dropna().astype(str).tolist()

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


def _bool_env(name: str, default: str = '0') -> bool:
    return str(os.getenv(name, default)).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def _get_lightgpt_settings() -> tuple[str, str]:
    """Return (base_url, api_key) for LightGPT.

    LightGPT lives on the same Nostradamus API base URL.
    We reuse the existing Nostradamus settings + API key.
    """

    base_url = (st.session_state.get('nostradamus_api_base_url') or '').strip()
    api_key = (st.session_state.get('nostradamus_api_key') or os.getenv('NOSTRADAMUS_API_KEY') or os.getenv('API_KEY') or '').strip()
    return base_url, api_key


def _load_item_features(project: str) -> pd.DataFrame:
    con = get_connection()
    df = con.execute(
        """
        SELECT project, item_id, name, item_type, flavour, size
        FROM item_features
        WHERE project = ?
        ORDER BY item_id
        """,
        [project],
    ).fetchdf()
    con.close()

    # Normalize item_id to str
    if not df.empty and 'item_id' in df.columns:
        df['item_id'] = df['item_id'].astype(str)
    return df


@st.cache_data(ttl=60)
def _get_item_name(project: str, item_id: str) -> str:
    """Return item_features.name for (project, item_id) or '' if missing."""
    con = get_connection()
    try:
        df = con.execute(
            """
            SELECT name
            FROM item_features
            WHERE project = ? AND item_id = ?
            LIMIT 1
            """,
            [project, str(item_id)],
        ).fetchdf()
    except Exception:
        return ""
    finally:
        con.close()

    if df is None or df.empty:
        return ""
    v = df.iloc[0].get('name')
    if pd.isna(v):
        return ""
    return str(v)


@st.cache_data(ttl=60)
def _get_item_features_item_ids(project: str) -> set[str]:
    try:
        df = _load_item_features(project)
    except Exception:
        return set()
    if df is None or df.empty or 'item_id' not in df.columns:
        return set()
    return set(df['item_id'].dropna().astype(str).tolist())


def _load_item_features_all() -> pd.DataFrame:
    con = get_connection()
    df = con.execute(
        """
        SELECT project, item_id, name, item_type, flavour, size
        FROM item_features
        ORDER BY project, item_id
        """
    ).fetchdf()
    con.close()

    if not df.empty:
        df['project'] = df['project'].astype(str)
        df['item_id'] = df['item_id'].astype(str)
    return df


@st.cache_data
def _get_item_features_history_date_range() -> tuple[date | None, date | None]:
    con = get_connection()
    df = con.execute(
        """
        SELECT MIN(s.sale_date) AS min_date, MAX(s.sale_date) AS max_date
        FROM sales_history s
        JOIN item_features f
          ON s.project = f.project
         AND s.item_id = f.item_id
        """
    ).fetchdf()
    con.close()
    if df.empty or df.iloc[0].isna().any():
        return None, None
    return pd.to_datetime(df.iloc[0]['min_date']).date(), pd.to_datetime(df.iloc[0]['max_date']).date()


def _lightgpt_external_item_id(project: str, item_id: str) -> str:
    return f"{project}::{item_id}"


def _build_lightgpt_payload(
    *,
    features_df: pd.DataFrame,
    hist_df: pd.DataFrame,
    forecast_periods: int,

) -> tuple[dict, dict[str, date], dict[str, tuple[str, str]]]:
    """Build LightGPT request payload (OpenAPI LightGPTForecastRequest).

    We send:
    - sim_input_his: list of {item_id, actual_sale, day} at monthly frequency (month-start)
    - item_attributes: list of {item_id, name, item_type, flavour, size}
    - forecast_periods: int
    - freq: 'M'
    """

    if hist_df.empty:
        return {}, {}, {}

    df_hist = hist_df.copy()
    df_hist['sale_date'] = pd.to_datetime(df_hist['sale_date'], errors='coerce')
    df_hist = df_hist[df_hist['sale_date'].notna()].copy()
    if df_hist.empty:
        return {}, {}, {}

    # Monthly aggregate to month-start
    df_hist['sale_date'] = df_hist['sale_date'].dt.to_period('M').dt.to_timestamp(how='start')
    df_hist['project'] = df_hist['project'].astype(str)
    df_hist['item_id'] = df_hist['item_id'].astype(str)
    df_hist = (
        df_hist.groupby(['project', 'item_id', 'sale_date'], as_index=False)['sales']
        .sum()
        .sort_values(['project', 'item_id', 'sale_date'])
    )
    # LightGPT tends to expect non-negative demand signals; also avoid NaNs.
    df_hist['sales'] = pd.to_numeric(df_hist['sales'], errors='coerce').fillna(0.0)
    df_hist.loc[df_hist['sales'] < 0, 'sales'] = 0.0
    df_hist['sale_date'] = df_hist['sale_date'].dt.date

    # Disambiguate across projects
    df_hist['ext_item_id'] = df_hist.apply(
        lambda r: _lightgpt_external_item_id(str(r['project']), str(r['item_id'])),
        axis=1,
    )

    # as_of per item = last observed month in (monthly aggregated) history
    as_of_map: dict[str, date] = df_hist.groupby('ext_item_id')['sale_date'].max().to_dict() if not df_hist.empty else {}
    id_map: dict[str, tuple[str, str]] = (
        df_hist.groupby('ext_item_id')[['project', 'item_id']].first().apply(tuple, axis=1).to_dict()
        if not df_hist.empty else {}
    )

    # IMPORTANT: avoid creating duplicate 'item_id' columns by selecting ext_item_id only.
    df_for_sim = df_hist[['ext_item_id', 'sale_date', 'sales']].copy()
    df_for_sim = df_for_sim.rename(columns={'ext_item_id': 'item_id'})
    sim_input_his = _format_sim_input(df_for_sim)

    feat = features_df.copy()
    feat['project'] = feat['project'].astype(str)
    feat['item_id'] = feat['item_id'].astype(str)
    item_attributes = []
    for _, r in feat.iterrows():
        ext_id = _lightgpt_external_item_id(str(r.get('project')), str(r.get('item_id')))
        item_attributes.append(
            {
                'item_id': ext_id,
                'name': None if pd.isna(r.get('name')) else str(r.get('name')),
                'item_type': None if pd.isna(r.get('item_type')) else str(r.get('item_type')),
                'flavour': None if pd.isna(r.get('flavour')) else str(r.get('flavour')),
                'size': None if pd.isna(r.get('size')) else str(r.get('size')),
            }
        )

    payload = {
        'sim_input_his': sim_input_his,
        'item_attributes': item_attributes,
        'forecast_periods': int(forecast_periods),
        'forecast_type': 'batch',
        # LightGPT API expects monthly as 'M' (not 'MS').
        # We aggregate to month-start internally, but still request monthly periods.
        'freq': 'M',
    }
    return payload, as_of_map, id_map


def _parse_lightgpt_response(
    resp: dict,
    *,
    run_tag: str,
    as_of_map: dict[str, date],
    id_map: dict[str, tuple[str, str]],
    forecast_periods: int,
) -> pd.DataFrame:
    """Parse LightGPT response into the shared `forecasts` table shape.

    Supported response shapes (best-effort):
    - {"forecasts": [{"item_id":..., "forecast":[...], "forecast_dates":[...]}]}
    - {"results":  [{"item_id":..., "forecast":[...], "forecast_dates":[...]}]}
    - {"items":    [{"item_id":..., "forecast":[...]}]}  (dates auto-generated from as_of_map)
    - {"forecasts": [{"item_id":..., "day": "YYYY-MM-DD...", "forecast": 123.4}, ...]} (row-wise)
    """

    if not isinstance(resp, dict):
        return pd.DataFrame([])

    forecasts = resp.get('forecasts') or resp.get('results') or resp.get('items') or []
    if not isinstance(forecasts, list):
        return pd.DataFrame([])

    rows: list[dict] = []
    method = f"lightgpt@{run_tag}"

    for it in forecasts:
        if not isinstance(it, dict):
            continue
        ext_item_id = it.get('item_id') or it.get('item') or it.get('sku')
        if ext_item_id is None:
            continue
        ext_item_id = str(ext_item_id)

        mapped = id_map.get(ext_item_id)
        if not mapped and '::' in ext_item_id:
            p, i = ext_item_id.split('::', 1)
            mapped = (p, i)
        if not mapped:
            continue
        project, item_id = mapped

        # Row-wise shape: each element is a single forecast row with a date field.
        # Example: {"item_id": "Project::123", "day": "2024-11-02T00:00:00", "forecast": 807.75}
        date_field = (
            it.get('forecast_date')
            or it.get('day')
            or it.get('date')
            or it.get('ds')
            or it.get('timestamp')
        )
        value_field = it.get('forecast')
        if date_field is not None and isinstance(value_field, (int, float, str)):
            try:
                rows.append(
                    {
                        'project': project,
                        'forecast_method': method,
                        'item_id': item_id,
                        'forecast_date': pd.to_datetime(date_field).date(),
                        'forecast': float(value_field),
                    }
                )
            except Exception:
                pass
            continue

        vals = it.get('forecast') or it.get('forecasts') or it.get('values')
        dates = it.get('forecast_dates') or it.get('dates')

        if not isinstance(vals, list):
            continue

        # If no dates returned, generate monthly dates from the item's as_of date
        if not isinstance(dates, list) or len(dates) != len(vals):
            as_of = as_of_map.get(ext_item_id)
            if as_of is None:
                continue
            gen_dates = _generate_forecast_dates(as_of, int(len(vals) or forecast_periods), 'M')
            dates = [d.isoformat() for d in gen_dates[: len(vals)]]

        for d, v in zip(dates, vals):
            try:
                rows.append(
                    {
                        'project': project,
                        'forecast_method': method,
                        'item_id': item_id,
                        'forecast_date': pd.to_datetime(d).date(),
                        'forecast': float(v),
                    }
                )
            except Exception:
                continue

    return pd.DataFrame(rows)


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

    with st.sidebar.expander('Settings', expanded=False):
        st.text_input(
            'Nostradamus API base URL',
            key='nostradamus_api_base_url',
            help='Base URL for forecast jobs, e.g. https://api.nostradamus-api.com',
        )
        st.text_input(
            'Nostradamus / LightGPT API key (optional)',
            type='password',
            key='nostradamus_api_key',
            help='Used for Nostradamus forecast endpoints and LightGPT; sent as X-API-Key header (leave blank if not required).',
        )
        st.text_input(
            'TimeGPT API key (optional)',
            type='password',
            key='timegpt_api_key',
            help='Only used when Mode=timegpt; sent in the request payload (this is separate from the Nostradamus/LightGPT key).',
        )

        # LightGPT uses the same Nostradamus base URL + key; feature-flag only.

    projects = get_projects()
    if not projects:
        st.warning("No data found. Import CSVs first using `python ingest.py`.")
        return

    _sync_project_widget('app_project', projects)

    def _on_app_project_change() -> None:
        st.session_state['active_project'] = st.session_state.get('app_project')

    project = st.sidebar.selectbox(
        "Project",
        projects,
        key='app_project',
        on_change=_on_app_project_change,
    )

    # Rendered later (below prev/next buttons), but we still need its value
    # here to filter the item list deterministically.
    only_item_features = bool(st.session_state.get('main_only_item_features', False))

    items = get_items(project)
    if not items:
        st.warning("No items for this project.")
        return

    if only_item_features:
        allowed = _get_item_features_item_ids(project)
        if not allowed:
            st.warning("No rows found in item_features for this project.")
            return
        items = [i for i in items if str(i) in allowed]
        if not items:
            st.warning("No items left after filtering to item_features.")
            return

    _ensure_single_choice_state('main_item_id', items)

    def _select_prev_item() -> None:
        cur = st.session_state.get('main_item_id')
        try:
            idx = items.index(cur)
        except Exception:
            idx = 0
        st.session_state['main_item_id'] = items[max(0, idx - 1)]

    def _select_next_item() -> None:
        cur = st.session_state.get('main_item_id')
        try:
            idx = items.index(cur)
        except Exception:
            idx = 0
        st.session_state['main_item_id'] = items[min(len(items) - 1, idx + 1)]

    item_id = st.sidebar.selectbox("Item", items, key='main_item_id')

    item_name = _get_item_name(project, item_id)
    name_suffix = f" - {item_name}" if item_name else ""
    header_line = f"Project: {project} - Item: {item_id}{name_suffix}"
    st.markdown(
        f"<div style='font-size: 1.3rem; line-height: 1.6; margin-top: -0.25rem;'>"
        f"{html.escape(header_line)}"
        f"</div>",
        unsafe_allow_html=True,
    )

    try:
        _cur_idx = items.index(st.session_state.get('main_item_id'))
    except Exception:
        _cur_idx = 0

    c_prev, c_next = st.sidebar.columns(2)
    c_prev.button("Prev item", on_click=_select_prev_item, disabled=_cur_idx <= 0, use_container_width=True)
    c_next.button(
        "Next item",
        on_click=_select_next_item,
        disabled=_cur_idx >= (len(items) - 1),
        use_container_width=True,
    )

    st.sidebar.checkbox(
        "Only selected items",
        value=only_item_features,
        key='main_only_item_features',
        help="When enabled, only show items that exist in the item_features table for this project.",
    )

    # Viewing: always show all methods; legend click toggles visibility.
    methods = get_forecast_methods(project, item_id)
    selected_methods = methods

    st.sidebar.markdown("---")

    # --- LightGPT batch forecast section (feature-flagged) ---
    if _bool_env('ENABLE_LIGHTGPT', '0'):
        if st.sidebar.button("Generate LightGPT forecasts (item_features)"):
            next_open = not bool(st.session_state.get('_show_lightgpt_forecast', False))
            st.session_state['_show_lightgpt_forecast'] = next_open
            if next_open:
                st.session_state['_show_project_forecast'] = False
                st.session_state['_show_item_forecast'] = False

        if st.session_state.get('_show_lightgpt_forecast'):
            with st.sidebar.container():
                st.markdown("**LightGPT batch parameters**")
                with st.form('lightgpt_form'):
                    min_lg, max_lg = _get_item_features_history_date_range()
                    max_eff = _effective_as_of_for_monthly(max_lg) if max_lg else None
                    lg_periods = st.number_input(
                        'Forecast periods (months)',
                        min_value=1,
                        max_value=120,
                        value=12,
                        key='lightgpt_forecast_periods',
                    )
                    lg_as_of = st.date_input(
                        'Forecast as-of date',
                        value=max_eff if max_eff else datetime.now().date(),
                        min_value=min_lg,
                        max_value=max_lg,
                        key='lightgpt_as_of_date',
                        help='History is truncated to this date; monthly forecasts start after this date (month-aligned).',
                    )
                    lg_timeout = st.number_input(
                        'Request timeout (s)',
                        min_value=10,
                        max_value=600,
                        value=60,
                        key='lightgpt_timeout_seconds',
                    )
                    submitted_lg = st.form_submit_button('Run LightGPT batch')

                if submitted_lg:
                    st.session_state['_show_lightgpt_forecast'] = False
                    st.session_state['_lightgpt_forecast_request'] = {
                        'forecast_periods': int(lg_periods),
                        'as_of_date': lg_as_of.isoformat() if lg_as_of else None,
                        'timeout_seconds': int(lg_timeout),
                    }
                    st.rerun()

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
                if 'proj_run_mode' not in st.session_state:
                    st.session_state['proj_run_mode'] = 'local'
                run_mode = st.selectbox("Mode", options=["local", "timegpt"], index=0, key='proj_run_mode')
                local_model = st.selectbox(
                    "Local model",
                    options=[
                        'auto_model','auto_arima','auto_ets','naive','seasonal_naive','croston_optimized','adida','theta','optimized_theta','auto_ces'
                    ],
                    index=0,
                    key='proj_local_model',
                )
                if 'proj_freq' not in st.session_state:
                    st.session_state['proj_freq'] = 'M'
                freq = st.selectbox("Frequency", options=['D','M','W','H','Q','Y'], index=1, key='proj_freq')
                aggregate_monthly = st.checkbox(
                    "Aggregate to monthly totals",
                    value=True,
                    key='proj_aggregate_monthly',
                    help="If enabled, daily sales are summed per month and sent with dates set to the 1st of the month. Frequency will be sent as M.",
                )
                season_length = st.number_input("Season length", min_value=1, max_value=365, value=12, key='proj_season_length')
                forecast_periods = st.number_input("Forecast periods", min_value=1, max_value=365, value=12, key='proj_forecast_periods')
                quantiles_text = st.text_input("Quantiles (comma-separated, for TimeGPT)", value="", key='proj_quantiles_text')
                webhook_url = st.text_input("Webhook URL (optional)", key='proj_webhook_url')
                wait_for_completion = st.checkbox("Wait for completion (poll status)", value=True, key='proj_wait_for_completion')
                poll_timeout_seconds = st.number_input(
                    "Polling timeout (s)",
                    min_value=60,
                    max_value=24 * 60 * 60,
                    value=1800,
                    key='proj_poll_timeout_seconds',
                )
                run_in_batches = st.checkbox("Run in batches", value=True, key='proj_run_in_batches')
                batch_size_items = st.number_input("Batch size (items per job)", min_value=1, max_value=5000, value=500, key='proj_batch_size_items')
                min_d, max_d = get_history_date_range(project)
                if min_d and max_d:
                    # Sticky default: if the widget key disappeared (panel closed), restore last choice.
                    if 'proj_as_of_date' not in st.session_state and st.session_state.get('proj_as_of_date_last') is not None:
                        st.session_state['proj_as_of_date'] = st.session_state.get('proj_as_of_date_last')

                    prev = st.session_state.get('proj_as_of_date')
                    if prev is not None:
                        try:
                            prev_d = pd.to_datetime(prev).date() if not isinstance(prev, date) else prev
                            if prev_d < min_d:
                                st.session_state['proj_as_of_date'] = min_d
                            elif prev_d > max_d:
                                st.session_state['proj_as_of_date'] = max_d
                        except Exception:
                            st.session_state['proj_as_of_date'] = max_d
                as_of_date_project = st.date_input(
                    "Forecast as-of date",
                    value=max_d if max_d else datetime.now().date(),
                    min_value=min_d,
                    max_value=max_d,
                    key='proj_as_of_date',
                    help="Only history up to this date is used; forecast starts the day after (freq-aligned).",
                )
                _ensure_multi_choice_state('proj_item_selection', items, default=[])
                item_selection = st.multiselect(
                    "Items to include (leave empty = all items in project)",
                    options=items,
                    default=[],
                    key='proj_item_selection',
                )
                submitted = st.form_submit_button("Batch run")

            if submitted:
                st.session_state['_show_project_forecast'] = False
                # Remember as-of date so it stays sticky when reopening the panel.
                st.session_state['proj_as_of_date_last'] = as_of_date_project
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
                if 'item_run_mode' not in st.session_state:
                    st.session_state['item_run_mode'] = 'local'
                run_mode_item = st.selectbox("Mode", options=["local", "timegpt"], index=0, key='item_run_mode')
                local_model_item = st.selectbox(
                    "Local model",
                    options=[
                        'auto_model','auto_arima','auto_ets','naive','seasonal_naive','croston_optimized','adida','theta','optimized_theta','auto_ces'
                    ],
                    index=0,
                    key='item_local_model',
                )
                if 'item_freq' not in st.session_state:
                    st.session_state['item_freq'] = 'M'
                freq_item = st.selectbox("Frequency", options=['D','M','W','H','Q','Y'], index=1, key='item_freq')
                aggregate_monthly_item = st.checkbox(
                    "Aggregate to monthly totals",
                    value=True,
                    key='item_aggregate_monthly',
                    help="If enabled, daily sales are summed per month and sent with dates set to the 1st of the month. Frequency will be sent as M.",
                )
                season_length_item = st.number_input("Season length", min_value=1, max_value=365, value=12, key='item_season_length')
                forecast_periods_item = st.number_input("Forecast periods", min_value=1, max_value=365, value=12, key='item_forecast_periods')
                min_di, max_di = get_history_date_range(project, item_id=item_id)
                if min_di and max_di:
                    # Sticky default: if the widget key disappeared (panel closed), restore last choice.
                    if 'item_as_of_date' not in st.session_state and st.session_state.get('item_as_of_date_last') is not None:
                        st.session_state['item_as_of_date'] = st.session_state.get('item_as_of_date_last')

                    prev = st.session_state.get('item_as_of_date')
                    if prev is not None:
                        try:
                            prev_d = pd.to_datetime(prev).date() if not isinstance(prev, date) else prev
                            if prev_d < min_di:
                                st.session_state['item_as_of_date'] = min_di
                            elif prev_d > max_di:
                                st.session_state['item_as_of_date'] = max_di
                        except Exception:
                            st.session_state['item_as_of_date'] = max_di
                as_of_date_item = st.date_input(
                    "Forecast as-of date",
                    value=max_di if max_di else datetime.now().date(),
                    min_value=min_di,
                    max_value=max_di,
                    key='item_as_of_date',
                    help="Only history up to this date is used; forecast starts the day after (freq-aligned).",
                )
                quantiles_text_item = st.text_input("Quantiles (comma-separated, for TimeGPT)", value="", key='item_quantiles_text')
                timeout_seconds_item = st.number_input("Request timeout (s)", min_value=30, max_value=1200, value=300, key='item_timeout_seconds')
                submitted_item = st.form_submit_button("Single run")

            if submitted_item:
                st.session_state['_show_item_forecast'] = False
                # Remember as-of date so it stays sticky when reopening the panel.
                st.session_state['item_as_of_date_last'] = as_of_date_item
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
    lightgpt_req = st.session_state.pop('_lightgpt_forecast_request', None)
    if lightgpt_req:
        st.markdown('---')
        st.subheader('LightGPT batch run')

        run_tag = _run_suffix()
        st.caption(f"Run tag: @\"{run_tag}\" (saved as method lightgpt@{run_tag})")

        features_df = _load_item_features_all()
        if features_df.empty:
            st.error('No rows found in item_features.')
        else:
            con = get_connection()
            try:
                hist_df = con.execute(
                    """
                    SELECT s.project, s.item_id, s.sale_date, s.sales
                    FROM sales_history s
                    JOIN item_features f
                      ON s.project = f.project
                     AND s.item_id = f.item_id
                    ORDER BY s.project, s.item_id, s.sale_date
                    """
                ).fetchdf()
            finally:
                con.close()

            if hist_df.empty:
                st.error('No sales_history rows found for item_features items.')
            else:
                as_of_date = None
                if lightgpt_req.get('as_of_date'):
                    try:
                        as_of_date = pd.to_datetime(lightgpt_req['as_of_date']).date()
                    except Exception:
                        as_of_date = None

                if as_of_date is not None:
                    # For monthly runs, use a month-end aligned cutoff to avoid partial months.
                    eff_as_of = _effective_as_of_for_monthly(as_of_date)
                    hist_df, _ = _apply_as_of_cutoff(hist_df, eff_as_of)

                payload, as_of_map, id_map = _build_lightgpt_payload(
                    features_df=features_df,
                    hist_df=hist_df,
                    forecast_periods=int(lightgpt_req['forecast_periods']),
                )
                if not payload.get('sim_input_his'):
                    st.error('Could not build LightGPT payload (no usable histories).')
                else:
                    base_url, key = _get_lightgpt_settings()
                    base_url = _normalize_localhost_url(base_url)
                    if not base_url:
                        st.error('Nostradamus API base URL is not set.')
                    else:
                        if not key:
                            st.warning(
                                "No Nostradamus API key is set. We'll call LightGPT without X-API-Key. "
                                "If the hosted API returns an error, the response body will be shown to help diagnose it."
                            )
                        st.caption(f"Sending item_features batch to LightGPT (freq={payload.get('freq')!r}).")
                        try:
                            resp = call_lightgpt_batch(
                                base_url,
                                payload,
                                api_key=(key or None),
                                timeout_s=float(lightgpt_req['timeout_seconds']),
                            )
                        except Exception as e:
                            st.error(f'LightGPT request failed: {e}')
                            resp = None

                        if resp is not None:
                            with st.expander('LightGPT response', expanded=False):
                                st.json(resp)

                            df_new = _parse_lightgpt_response(
                                resp,
                                run_tag=run_tag,
                                as_of_map=as_of_map,
                                id_map=id_map,
                                forecast_periods=int(lightgpt_req['forecast_periods']),
                            )

                            if df_new.empty:
                                st.error('Could not parse any forecast rows from LightGPT response.')
                            else:
                                con = get_connection()
                                try:
                                    df_new = df_new.copy()
                                    con.register('forecasts_lightgpt_df', df_new)
                                    # Overwrite previous LightGPT runs for these items (base method lightgpt).
                                    con.execute(
                                        """
                                        DELETE FROM forecasts
                                        WHERE split_part(forecast_method, '@', 1) = 'lightgpt'
                                          AND (project, item_id) IN (
                                              SELECT DISTINCT project, item_id
                                              FROM forecasts_lightgpt_df
                                          )
                                        """
                                    )
                                    con.execute(
                                        """
                                        INSERT INTO forecasts (project, forecast_method, item_id, forecast_date, forecast)
                                        SELECT project, forecast_method, item_id, forecast_date, forecast
                                        FROM forecasts_lightgpt_df
                                        """
                                    )
                                finally:
                                    con.close()

                                st.success(f"Saved {len(df_new)} LightGPT forecast rows.")
                                st.cache_data.clear()
                                st.rerun()

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

    # (Header caption shown at top of page.)

    item_req = st.session_state.pop('_item_forecast_request', None)
    if item_req:
        st.markdown('---')
        st.subheader('Item forecast run')

        con = get_connection()
        try:
            df_hist = con.execute(
                """
                SELECT item_id, sale_date, sales
                FROM sales_history
                WHERE project = ?
                  AND item_id = ?
                ORDER BY item_id, sale_date
                """,
                [item_req['project'], item_req['item_id']],
            ).fetchdf()
        finally:
            con.close()

        if df_hist.empty:
            st.error("No sales history found for this item.")
        else:
            as_of_date_item = None
            if item_req.get('as_of_date'):
                try:
                    as_of_date_item = pd.to_datetime(item_req['as_of_date']).date()
                except Exception:
                    as_of_date_item = None

            if as_of_date_item is not None:
                df_hist, _ = _apply_as_of_cutoff(df_hist, as_of_date_item)
                if df_hist.empty:
                    st.error("No history rows left after applying as-of date.")
                    return

            if item_req['aggregate_monthly']:
                df_hist = _aggregate_monthly_sales(df_hist)

            sim_input_his = _format_sim_input(df_hist)
            if len(sim_input_his) < 5:
                st.warning("Less than 5 data points — some models may not work well.")

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

            if payload is None:
                return

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
    
    col1, col2, col3 = st.columns([0.16, 0.68, 0.16])
    
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
        key='main_show_monthly_history',
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
    # Pinned colors + legend toggling for History, History (Monthly), and methods.
    # History is always first in the legend.
    method_domain = get_project_forecast_methods(project)

    # Tableau20 palette (20 colors) for stable method mapping.
    tableau20 = [
        "#4E79A7",
        "#F28E2B",
        "#E15759",
        "#76B7B2",
        "#59A14F",
        "#EDC948",
        "#B07AA1",
        "#FF9DA7",
        "#9C755F",
        "#BAB0AC",
        "#A0CBE8",
        "#FFBE7D",
        "#FF9D9A",
        "#9CDED6",
        "#8CD17D",
        "#F1CE63",
        "#D4A6C8",
        "#FABFD2",
        "#D7B5A6",
        "#D3D3D3",
    ]

    method_colors = [tableau20[i % len(tableau20)] for i in range(len(method_domain))]
    method_color_map = dict(zip(method_domain, method_colors))

    present = set(combined_filtered['series'].dropna().astype(str).unique().tolist())
    include_monthly = bool(show_monthly_history) and ("History (Monthly)" in present)

    # Legend ordering: History, then History (Monthly) if enabled, then methods.
    present_methods = [m for m in method_domain if m in present]
    present_method_colors = [method_color_map[m] for m in present_methods]

    series_domain = ["History"] + (["History (Monthly)"] if include_monthly else []) + present_methods
    series_range = ["blue"] + (["lightblue"] if include_monthly else []) + present_method_colors

    legend_domain = [s for s in series_domain if s in present]

    series_sel = alt.selection_point(
        fields=["series"],
        bind="legend",
        toggle="true",
        value=[{"series": s} for s in legend_domain],
        empty="none",
    )

    chart = (
        alt.Chart(combined_filtered)
        .mark_line(point=True)
        .transform_filter(series_sel)
        .encode(
            x=alt.X("date:T"),
            y=alt.Y("value:Q"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=series_domain, range=series_range),
            ),
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

    if st.button(
        "Recalculate error metrics for selected item/methods",
        key="recalc_item_metrics_btn",
        help="Recomputes metrics from sales_history + forecasts for this item and overwrites matching rows in forecast_metrics.",
    ):
        from metrics import recompute_item_metrics

        with st.spinner("Recomputing metrics for this item…"):
            inserted = recompute_item_metrics(
                project=str(project),
                item_id=str(item_id),
                base_methods=list(selected_methods or []),
            )
        st.cache_data.clear()
        st.success(f"Recomputed metrics. Inserted {inserted} rows.")
        st.rerun()

    # Show data
    with st.expander("Show raw data"):
        st.write("**Sales history**")
        st.dataframe(history_filtered.sort_values("date"))
        st.write("**Forecasts**")
        st.dataframe(forecasts_filtered.sort_values("date"))

if __name__ == "__main__":
    main()
