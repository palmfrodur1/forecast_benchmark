#!/usr/bin/env python3
import argparse
import time
import json
from datetime import date, timedelta
from pathlib import Path
from db import get_connection
import pandas as pd
from timing_utils import log, start_timer


def _format_sim_input(df_history: pd.DataFrame) -> list:
    out = []
    if df_history.empty:
        return out
    df = df_history.copy()
    df['sale_date'] = pd.to_datetime(df['sale_date']).dt.date
    df = df.sort_values(['item_id', 'sale_date'])
    for _, row in df.iterrows():
        out.append({
            'item_id': row['item_id'],
            'actual_sale': float(row['sales']),
            'day': row['sale_date'].isoformat()
        })
    return out


def _forecast_start_after_as_of(as_of: date, freq: str) -> pd.Timestamp:
    ts = pd.Timestamp(as_of)
    f = str(freq or '').upper()
    if f.startswith('M'):
        return (ts + pd.offsets.MonthBegin(1)).normalize()
    return (ts + pd.Timedelta(days=1)).normalize()


def _generate_forecast_dates(as_of: date, periods: int, freq: str) -> list[date]:
    start = _forecast_start_after_as_of(as_of, freq)
    f = str(freq or '').upper()
    if f.startswith('M'):
        rng = pd.date_range(start=start, periods=int(periods), freq='MS')
    else:
        rng = pd.date_range(start=start, periods=int(periods), freq=freq or 'D')
    return [pd.Timestamp(d).date() for d in rng]


def _parse_nostradamus_response(
    resp: dict,
    project: str,
    fm_override: str | None = None,
    *,
    as_of_date: date | None = None,
    freq: str | None = None,
) -> pd.DataFrame:
    rows = []
    if not isinstance(resp, dict):
        return pd.DataFrame(rows)

    forecasts = resp.get('forecasts') or []
    top_model = resp.get('model')

    for it in forecasts:
        item_id = it.get('item_id') or it.get('item') or it.get('sku')
        model_used = it.get('model_used')
        if str(fm_override or '').strip().lower() == 'auto_model':
            resolved = model_used or top_model or 'unknown'
            fm = f"AM:{resolved}"
        else:
            fm = model_used or fm_override or top_model or 'unknown'

        vals = it.get('forecast') or it.get('forecasts') or it.get('values')
        dates = it.get('forecast_dates') or it.get('dates') or it.get('forecast_date')

        use_generated_dates = as_of_date is not None and bool(freq)

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

    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def _parse_as_of_date(value: str | None) -> date | None:
    if value is None:
        return None
    v = str(value).strip()
    if not v:
        return None
    return pd.to_datetime(v, errors='raise').date()


def _apply_as_of_cutoff(df_hist: pd.DataFrame, as_of: date | None) -> tuple[pd.DataFrame, date | None]:
    """Filter history to sale_date <= as_of.

    Returns (filtered_df, effective_as_of). If as_of is None, effective_as_of is
    the max available sale_date.
    """
    if df_hist is None or df_hist.empty:
        return df_hist, as_of

    df = df_hist.copy()
    df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
    df = df[df['sale_date'].notna()].copy()
    if df.empty:
        return df, as_of

    min_d = df['sale_date'].min().date()
    max_d = df['sale_date'].max().date()

    if as_of is None:
        as_of_eff = max_d
    else:
        as_of_eff = as_of
        if as_of_eff > max_d:
            # Can't backtest "from the future"; clamp to last observed date.
            as_of_eff = max_d
        if as_of_eff < min_d:
            raise ValueError(f"as_of_date={as_of_eff.isoformat()} is before first history date {min_d.isoformat()}")

    df = df[df['sale_date'].dt.date <= as_of_eff].copy()
    # Restore original date-like type expected elsewhere
    df['sale_date'] = df['sale_date'].dt.date
    return df, as_of_eff


def _effective_as_of_for_monthly(as_of: date | None) -> date | None:
    """For monthly frequency, avoid including partial months.

    If as_of isn't the last day of its month, use the previous month-end.
    """
    if as_of is None:
        return None
    ts = pd.Timestamp(as_of)
    month_end = (ts + pd.offsets.MonthEnd(0)).date()
    if as_of == month_end:
        return as_of
    return (ts - pd.offsets.MonthEnd(1)).date()
import requests
import pandas as pd


def local_seasonal_mean(df_hist, season_length, periods, freq, project, item_id, *, as_of_date: date | None = None):
    # df_hist expected columns: item_id, sale_date, sales
    df = df_hist.sort_values('sale_date')
    last_vals = df['sales'].dropna().astype(float).tolist()[-season_length:]
    if len(last_vals) == 0:
        mean_val = 0.0
    else:
        mean_val = float(pd.Series(last_vals).mean())

    # build forecast dates: start at the next period after as_of_date (freq-aligned)
    if as_of_date is None:
        as_of_eff = pd.to_datetime(df['sale_date'].iloc[-1]).date()
    else:
        as_of_eff = as_of_date
    dates = _generate_forecast_dates(as_of_eff, periods, freq)

    rows = []
    for d in dates:
        rows.append({
            'project': project,
            'forecast_method': 'seasonal_mean',
            'item_id': item_id,
            'forecast_date': d.isoformat(),
            'forecast': mean_val,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='SOG')
    parser.add_argument('--item-id', required=True)
    parser.add_argument(
        '--as-of-date',
        default=None,
        help='Cutoff date (YYYY-MM-DD). History after this date is ignored; forecast starts the day after (freq-aligned).',
    )
    parser.add_argument('--api-url', default='https://api.nostradamus-api.com/api/v1/forecast/generate_async')
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--retries', type=int, default=3)
    parser.add_argument(
        '--local-model',
        default='auto_arima',
        help='Local model name to send to the API (e.g. auto_model, auto_arima, auto_ces, ...).',
    )
    parser.add_argument('--season-length', type=int, default=12)
    parser.add_argument('--forecast-periods', type=int, default=12)
    parser.add_argument('--freq', default='M')
    parser.add_argument('--local-fallback', action='store_true')
    parser.add_argument('--local-only', action='store_true', help='Do not call remote API; compute local forecast only')
    args = parser.parse_args()

    run_start = start_timer()
    log("Starting item forecast run")

    PROJECT = args.project
    ITEM = args.item_id
    API_URL = args.api_url
    log(f"Project={PROJECT} item_id={ITEM} api_url={API_URL}", since=run_start)

    # fetch history for this item
    con = get_connection()
    sql = "SELECT item_id, sale_date, sales FROM sales_history WHERE project = ? AND item_id = ? ORDER BY sale_date"
    df_hist = con.execute(sql, [PROJECT, ITEM]).fetchdf()
    con.close()

    if df_hist.empty:
        log(f"No sales history found for project {PROJECT}, item {ITEM}. Aborting.", since=run_start)
        raise SystemExit(1)

    as_of_date = _parse_as_of_date(args.as_of_date)
    df_hist_cut, as_of_eff = _apply_as_of_cutoff(df_hist, as_of_date)
    if df_hist_cut.empty:
        log(f"History empty after applying as_of_date={args.as_of_date}. Aborting.", since=run_start)
        raise SystemExit(1)

    if as_of_eff is not None:
        log(f"Using as_of_date={as_of_eff.isoformat()} (history rows={len(df_hist_cut)}).", since=run_start)

    sim_input_his = _format_sim_input(df_hist_cut)
    log(f"Prepared sim_input_his with {len(sim_input_his)} rows.", since=run_start)
    payload = {
        'sim_input_his': sim_input_his,
        'forecast_periods': args.forecast_periods,
        'mode': 'local',
        'local_model': args.local_model,
        'season_length': args.season_length,
        'freq': args.freq,
    }

    resp_json = None
    if not args.local_only:
        for attempt in range(1, args.retries + 1):
            try:
                api_attempt_start = start_timer()
                log(f"Calling API (attempt {attempt}/{args.retries})...", since=run_start)
                r = requests.post(API_URL, json=payload, timeout=args.timeout)
                log(f"HTTP {r.status_code}", since=api_attempt_start)
                if r.status_code != 200:
                    print(r.text[:1000])
                    raise RuntimeError(f"HTTP {r.status_code}")
                resp_json = r.json()
                log("API call succeeded.", since=api_attempt_start)
                break
            except Exception as e:
                log(f"API request attempt {attempt} failed: {e}", since=run_start)
                if attempt < args.retries:
                    sleep = 5 * (2 ** (attempt - 1))
                    log(f"Retrying after {sleep}s...", since=run_start)
                    time.sleep(sleep)
                else:
                    log("All remote attempts failed.", since=run_start)
    else:
        log("Skipping remote API call (local-only mode)", since=run_start)

    out_path = Path(f"/tmp/nostradamus_response_item_{ITEM}.json")
    df_new = None
    if resp_json is not None:
        with out_path.open('w') as f:
            json.dump(resp_json, f, indent=2)
        log(f"Saved response to {out_path}", since=run_start)
        df_new = _parse_nostradamus_response(
            resp_json,
            project=PROJECT,
            fm_override=args.local_model,
            as_of_date=as_of_eff,
            freq=args.freq,
        )

    if (df_new is None or df_new.empty) and args.local_fallback:
        log("Falling back to local seasonal_mean method.", since=run_start)
        df_new = local_seasonal_mean(
            df_hist_cut,
            season_length=args.season_length,
            periods=args.forecast_periods,
            freq=args.freq,
            project=PROJECT,
            item_id=ITEM,
            as_of_date=as_of_eff,
        )

    if df_new is None or df_new.empty:
        log("No forecasts available (remote failed and no local fallback). Exiting.", since=run_start)
        raise SystemExit(1)

    # insert into DB (overwrite existing for this project+method+item)
    df_new = df_new.copy()
    fm_name = df_new['forecast_method'].iloc[0]
    db_start = start_timer()
    con = get_connection()
    try:
        con.register('forecasts_item_df', df_new)
        con.execute(
            'DELETE FROM forecasts WHERE project = ? AND item_id = ? AND forecast_method IN (SELECT DISTINCT forecast_method FROM forecasts_item_df)',
            [PROJECT, ITEM],
        )
        con.execute('INSERT INTO forecasts (project, forecast_method, item_id, forecast_date, forecast) SELECT project, forecast_method, item_id, forecast_date, forecast FROM forecasts_item_df')
    finally:
        con.close()

    log(f"Inserted {len(df_new)} forecast rows for {PROJECT}/{ITEM} (method={fm_name}, overwrote existing).", since=db_start)
    log("Item forecast run complete.", since=run_start)


if __name__ == '__main__':
    main()
