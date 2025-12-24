import argparse
import json
import time
import os
from math import ceil
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from db import get_connection
from app import _format_sim_input, _parse_nostradamus_response, _aggregate_monthly_sales
from client_nostradamus import submit_forecast_job, wait_for_job
from timing_utils import log, start_timer


def load_progress(path: Path):
    if path.exists():
        try:
            return set(json.loads(path.read_text()))
        except Exception:
            return set()
    return set()


def save_progress(path: Path, completed_batches):
    path.write_text(json.dumps(sorted(list(completed_batches)), indent=2))


def main():
    parser = argparse.ArgumentParser(description='Run project forecasts in batches (resumable).')
    parser.add_argument('--project', default='BMV_v1')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--forecast-periods', type=int, default=12)
    parser.add_argument('--local-model', default='auto_arima')
    parser.add_argument('--season-length', type=int, default=12)
    parser.add_argument('--freq', default='MS')
    parser.add_argument(
        '--aggregate-monthly',
        action='store_true',
        help='If set, sums daily sales per month and sets sale_date to month start (YYYY-MM-01); forces freq=MS in payload.',
    )
    parser.add_argument(
        '--base-url',
        default=os.getenv('NOSTRADAMUS_API_BASE_URL', 'https://api.nostradamus-api.com'),
        help='Nostradamus API base URL (include scheme; for local dev typically use http://localhost:8000)',
    )
    parser.add_argument('--webhook-url', default=None, help='Optional webhook URL for job completion callbacks')
    parser.add_argument(
        '--api-key',
        default=None,
        help='Optional API key (defaults to NOSTRADAMUS_API_KEY env var, falling back to API_KEY if unset)',
    )
    parser.add_argument('--submit-timeout', type=float, default=30.0, help='Seconds to wait for job submission request')
    parser.add_argument('--timeout', type=float, default=600.0, help='Total seconds to wait for a job to finish (polling)')
    parser.add_argument('--poll-interval', type=float, default=1.0, help='Polling interval seconds')
    parser.add_argument('--retries', type=int, default=3)
    parser.add_argument('--backoff-base', type=int, default=5, help='base seconds for exponential backoff')
    parser.add_argument('--progress-file', default='.nostradamus_progress.json', help='file to store completed batch indices')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    run_start = start_timer()
    log("Starting project forecast batch run")

    PROJECT = args.project
    BATCH_SIZE = args.batch_size
    FORECAST_PERIODS = args.forecast_periods
    LOCAL_MODEL = args.local_model
    SEASON_LENGTH = args.season_length
    FREQ = args.freq
    BASE_URL = args.base_url
    # Common local-dev footgun: Docker-published FastAPI is usually plain HTTP.
    parsed = urlparse(BASE_URL)
    if parsed.scheme == 'https' and (parsed.hostname in {'localhost', '127.0.0.1', '0.0.0.0'}):
        BASE_URL = urlunparse(parsed._replace(scheme='http'))
        print(f"Warning: changed base_url to {BASE_URL} (local hosts typically don't use TLS).")
    WEBHOOK_URL = args.webhook_url
    API_KEY = args.api_key or os.getenv('NOSTRADAMUS_API_KEY') or os.getenv('API_KEY')
    SUBMIT_TIMEOUT = args.submit_timeout
    TIMEOUT = args.timeout
    POLL_INTERVAL = args.poll_interval
    RETRIES = args.retries
    BACKOFF_BASE = args.backoff_base

    progress_path = Path(args.progress_file)
    completed = load_progress(progress_path)

    con = get_connection()
    items_df = con.execute("SELECT DISTINCT item_id FROM sales_history WHERE project = ? ORDER BY item_id", [PROJECT]).fetchdf()
    con.close()

    if items_df.empty:
        print(f"No items found for project {PROJECT}. Aborting.")
        raise SystemExit(1)

    item_list = items_df['item_id'].tolist()
    num_items = len(item_list)
    num_batches = ceil(num_items / BATCH_SIZE)
    log(f"Project {PROJECT}: {num_items} items, will run in {num_batches} batches (batch_size={BATCH_SIZE}).", since=run_start)

    for batch_idx in range(num_batches):
        batch_no = batch_idx + 1
        if batch_no in completed:
            log(f"Skipping batch {batch_no} (already completed according to {progress_path}).", since=run_start)
            continue

        batch_start = start_timer()

        start_idx = batch_idx * BATCH_SIZE
        batch_items = item_list[start_idx:start_idx + BATCH_SIZE]
        log(f"Batch {batch_no}/{num_batches} start: items {start_idx + 1}-{start_idx + len(batch_items)}")

        # Fetch history for these items
        placeholders = ','.join(['?'] * len(batch_items))
        sql = f"SELECT item_id, sale_date, sales FROM sales_history WHERE project = ? AND item_id IN ({placeholders}) ORDER BY item_id, sale_date"
        params = [PROJECT, *batch_items]
        con = get_connection()
        df_hist = con.execute(sql, params).fetchdf()
        con.close()

        if df_hist.empty:
            log("No history rows for this batch; marking completed and skipping.", since=batch_start)
            completed.add(batch_no)
            save_progress(progress_path, completed)
            continue

        if args.aggregate_monthly:
            df_hist = _aggregate_monthly_sales(df_hist)
        sim_input_his = _format_sim_input(df_hist)
        log(f"Prepared sim_input_his with {len(sim_input_his)} rows.", since=batch_start)

        payload = {
            'sim_input_his': sim_input_his,
            'forecast_periods': FORECAST_PERIODS,
            'mode': 'local',
            'local_model': LOCAL_MODEL,
            'season_length': SEASON_LENGTH,
            'freq': 'MS' if args.aggregate_monthly else FREQ,
        }

        if args.dry_run:
            log("Dry run - skipping API call and DB insert.", since=batch_start)
            completed.add(batch_no)
            save_progress(progress_path, completed)
            continue

        # Attempt job submission + polling with retries and exponential backoff
        resp_json = None
        for attempt in range(1, RETRIES + 1):
            try:
                api_attempt_start = start_timer()
                log(f"Submitting job (attempt {attempt}/{RETRIES})...", since=batch_start)
                job = submit_forecast_job(
                    BASE_URL,
                    payload,
                    webhook_url=WEBHOOK_URL,
                    api_key=API_KEY,
                    timeout_s=SUBMIT_TIMEOUT,
                )
                job_id = job.get('job_id')
                if not job_id:
                    raise RuntimeError(f"No job_id returned: {job}")
                log(f"Submitted job_id={job_id}. Polling status...", since=api_attempt_start)
                job_final = wait_for_job(
                    BASE_URL,
                    job_id,
                    api_key=API_KEY,
                    timeout_total_s=float(TIMEOUT),
                    poll_interval_s=float(POLL_INTERVAL),
                )
                status = job_final.get('status')
                if status != 'finished':
                    raise RuntimeError(f"Job ended with status={status}, error={job_final.get('error')}")

                resp_json = job_final.get('result') or {}
                log("Job finished successfully.", since=api_attempt_start)
                break
            except Exception as e:
                log(f"API request failed on attempt {attempt}: {e}", since=batch_start)
                if attempt < RETRIES:
                    sleep_sec = BACKOFF_BASE * (2 ** (attempt - 1))
                    log(f"Retrying after {sleep_sec}s backoff...", since=batch_start)
                    time.sleep(sleep_sec)
                else:
                    log("Giving up on this batch after retries.", since=batch_start)

        # Save raw response if present
        out_path = Path(f"/tmp/nostradamus_response_batch_{batch_no}.json")
        if resp_json is not None:
            with out_path.open('w') as f:
                json.dump(resp_json, f, indent=2)
            log(f"Saved raw response to {out_path}", since=batch_start)
        else:
            log("No JSON response to save for this batch.", since=batch_start)
            continue

        # Parse into dataframe
        df_new = _parse_nostradamus_response(resp_json, project=PROJECT, fm_override=LOCAL_MODEL)
        if df_new.empty:
            log("Parser returned no rows for this batch. Inspect raw response.", since=batch_start)
            continue

        run_tag = time.strftime('%Y%m%d_%H%M%S')
        df_new = df_new.copy()
        df_new['forecast_method'] = df_new['forecast_method'].astype(str) + '@' + run_tag

        # Insert into DB: keep each run distinct by tagging forecast_method
        con = get_connection()
        try:
            con.register('forecasts_api_df', df_new)
            con.execute('INSERT INTO forecasts (project, forecast_method, item_id, forecast_date, forecast) SELECT project, forecast_method, item_id, forecast_date, forecast FROM forecasts_api_df')
        finally:
            con.close()

        log(
            f"Batch {batch_no}/{num_batches} complete: inserted {len(df_new)} rows (tagged @{run_tag}).",
            since=batch_start,
        )

        # mark completed and save progress
        completed.add(batch_no)
        save_progress(progress_path, completed)

    log('All batches processed.', since=run_start)


if __name__ == '__main__':
    main()
