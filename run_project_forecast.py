from app import _format_sim_input, _parse_nostradamus_response
from client_nostradamus import submit_forecast_job, wait_for_job
from db import get_connection
import json, time
from timing_utils import log, start_timer

PROJECT = 'bmv_v1'
FORECAST_PERIODS = 12
LOCAL_MODEL = 'auto_arima'
SEASON_LENGTH = 12
FREQ = 'M'

run_start = start_timer()
log(f"Preparing forecast run for project={PROJECT} (model={LOCAL_MODEL}, periods={FORECAST_PERIODS})")
con = get_connection()
df_hist = con.execute("SELECT item_id, sale_date, sales FROM sales_history WHERE project = ? ORDER BY item_id, sale_date", [PROJECT]).fetchdf()
con.close()

if df_hist.empty:
    print(f"No sales history found for project {PROJECT}. Aborting.")
    raise SystemExit(1)

n_rows = len(df_hist)
unique_items = df_hist['item_id'].nunique() if 'item_id' in df_hist.columns else 'unknown'
log(f"Found {n_rows} history rows across {unique_items} items. Formatting payload...", since=run_start)

sim_input = _format_sim_input(df_hist)
log(f"Formatted sim_input_his with {len(sim_input)} rows (should equal history rows).", since=run_start)

payload = {
    'sim_input_his': sim_input,
    'forecast_periods': FORECAST_PERIODS,
    'mode': 'local',
    'local_model': LOCAL_MODEL,
    'season_length': SEASON_LENGTH,
    'freq': FREQ,
}

log('Submitting job to Nostradamus API (generate_job) and polling for result...', since=run_start)
api_start = start_timer()
try:
    ##base_url = 'https://localhost:8000'
    base_url = 'https://api.nostradamus-api.com'

    resp = submit_forecast_job(base_url, payload)
    log(f'Job submitted. job_id={resp.get("job_id")}', since=api_start)
    job_id = resp.get('job_id')
    if not job_id:
        raise SystemExit('No job_id returned by generate_job endpoint')
    job = wait_for_job(base_url, job_id, timeout_total_s=600, poll_interval_s=1.0)
    if job.get('status') != 'finished':
        raise SystemExit(f"Job ended with status: {job.get('status')}")
    resp = job.get('result') or {}
    log('Job finished, retrieved result payload', since=api_start)
except Exception as e:
    log(f'Job submission or polling failed: {e}', since=api_start)
    raise

# Save raw response for inspection
with open('/tmp/nostradamus_response.json', 'w') as f:
    json.dump(resp, f, indent=2)
log('Saved raw response to /tmp/nostradamus_response.json', since=run_start)

# Parse into DataFrame ready for DB
df_new = _parse_nostradamus_response(resp, project=PROJECT, fm_override=LOCAL_MODEL)
if df_new.empty:
    print('Could not parse forecasts from API response. Inspect /tmp/nostradamus_response.json')
    raise SystemExit(2)

log(f'Parsed {len(df_new)} forecast rows. Inserting into DB...', since=run_start)
con = get_connection()
run_tag = time.strftime('%Y%m%d_%H%M%S')
df_new = df_new.copy()
df_new['forecast_method'] = df_new['forecast_method'].astype(str) + '@' + run_tag
con.register('forecasts_api_df', df_new)
con.execute('INSERT INTO forecasts (project, forecast_method, item_id, forecast_date, forecast) SELECT project, forecast_method, item_id, forecast_date, forecast FROM forecasts_api_df')
con.close()
log(f'Inserted {len(df_new)} rows for project={PROJECT} (tagged @{run_tag}).', since=run_start)
log('Project forecast run complete.', since=run_start)
