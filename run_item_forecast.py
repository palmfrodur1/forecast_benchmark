#!/usr/bin/env python3
import argparse
import time
import json
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


def _parse_nostradamus_response(resp: dict, project: str, fm_override: str | None = None) -> pd.DataFrame:
    rows = []
    if not isinstance(resp, dict):
        return pd.DataFrame(rows)

    forecasts = resp.get('forecasts') or []
    top_model = resp.get('model')

    for it in forecasts:
        item_id = it.get('item_id') or it.get('item') or it.get('sku')
        fm = it.get('model_used') or top_model or fm_override or 'unknown'

        vals = it.get('forecast') or it.get('forecasts') or it.get('values')
        dates = it.get('forecast_dates') or it.get('dates') or it.get('forecast_date')

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
import requests
import pandas as pd


def local_seasonal_mean(df_hist, season_length, periods, freq, project, item_id):
    # df_hist expected columns: item_id, sale_date, sales
    df = df_hist.sort_values('sale_date')
    last_vals = df['sales'].dropna().astype(float).tolist()[-season_length:]
    if len(last_vals) == 0:
        mean_val = 0.0
    else:
        mean_val = float(pd.Series(last_vals).mean())

    # build forecast dates: start at next period after last sale_date
    last_date = pd.to_datetime(df['sale_date'].iloc[-1])
    # ensure period start: use pd.date_range with freq
    start = (last_date + pd.tseries.offsets.DateOffset(months=1)).to_period('M').to_timestamp()
    dates = pd.date_range(start=start, periods=periods, freq=freq)

    rows = []
    for d in dates:
        rows.append({
            'project': project,
            'forecast_method': 'seasonal_mean',
            'item_id': item_id,
            'forecast_date': d.date().isoformat(),
            'forecast': mean_val,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='SOG')
    parser.add_argument('--item-id', required=True)
    parser.add_argument('--api-url', default='https://api.nostradamus-api.com/api/v1/forecast/generate_async')
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--retries', type=int, default=3)
    parser.add_argument('--season-length', type=int, default=12)
    parser.add_argument('--forecast-periods', type=int, default=12)
    parser.add_argument('--freq', default='MS')
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

    sim_input_his = _format_sim_input(df_hist)
    log(f"Prepared sim_input_his with {len(sim_input_his)} rows.", since=run_start)
    payload = {
        'sim_input_his': sim_input_his,
        'forecast_periods': args.forecast_periods,
        'mode': 'local',
        'local_model': 'auto_arima',
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
        df_new = _parse_nostradamus_response(resp_json, project=PROJECT, fm_override='auto_arima')

    if (df_new is None or df_new.empty) and args.local_fallback:
        log("Falling back to local seasonal_mean method.", since=run_start)
        df_new = local_seasonal_mean(df_hist, season_length=args.season_length, periods=args.forecast_periods, freq=args.freq, project=PROJECT, item_id=ITEM)

    if df_new is None or df_new.empty:
        log("No forecasts available (remote failed and no local fallback). Exiting.", since=run_start)
        raise SystemExit(1)

    # insert into DB (delete existing for this project+method+item)
    run_tag = time.strftime('%Y%m%d_%H%M%S')
    df_new = df_new.copy()
    df_new['forecast_method'] = df_new['forecast_method'].astype(str) + '@' + run_tag
    fm_name = df_new['forecast_method'].iloc[0]
    db_start = start_timer()
    con = get_connection()
    try:
        con.register('forecasts_item_df', df_new)
        con.execute('INSERT INTO forecasts (project, forecast_method, item_id, forecast_date, forecast) SELECT project, forecast_method, item_id, forecast_date, forecast FROM forecasts_item_df')
    finally:
        con.close()

    log(f"Inserted {len(df_new)} forecast rows for {PROJECT}/{ITEM} (method={fm_name}).", since=db_start)
    log("Item forecast run complete.", since=run_start)


if __name__ == '__main__':
    main()
