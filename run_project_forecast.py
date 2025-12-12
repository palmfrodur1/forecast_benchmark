from app import _format_sim_input, _call_forecast_api, _parse_nostradamus_response
from db import get_connection
import json, time

PROJECT = 'bmv_v1'
FORECAST_PERIODS = 12
LOCAL_MODEL = 'auto_arima'
SEASON_LENGTH = 12
FREQ = 'MS'

print(f"Preparing forecast run for project={PROJECT} (model={LOCAL_MODEL}, periods={FORECAST_PERIODS})")
con = get_connection()
df_hist = con.execute("SELECT item_id, sale_date, sales FROM sales_history WHERE project = ? ORDER BY item_id, sale_date", [PROJECT]).fetchdf()
con.close()

if df_hist.empty:
    print(f"No sales history found for project {PROJECT}. Aborting.")
    raise SystemExit(1)

n_rows = len(df_hist)
unique_items = df_hist['item_id'].nunique() if 'item_id' in df_hist.columns else 'unknown'
print(f"Found {n_rows} history rows across {unique_items} items. Formatting payload...")

sim_input = _format_sim_input(df_hist)
print(f"Formatted sim_input_his with {len(sim_input)} rows (should equal history rows).")

payload = {
    'sim_input_his': sim_input,
    'forecast_periods': FORECAST_PERIODS,
    'mode': 'local',
    'local_model': LOCAL_MODEL,
    'season_length': SEASON_LENGTH,
    'freq': FREQ,
}

print('Sending payload to Nostradamus API...')
start = time.time()
try:
    resp = _call_forecast_api(payload)
    duration = time.time() - start
    print(f'API call completed in {duration:.1f}s')
except Exception as e:
    print('API call failed:', e)
    raise

# Save raw response for inspection
with open('/tmp/nostradamus_response.json', 'w') as f:
    json.dump(resp, f, indent=2)
print('Saved raw response to /tmp/nostradamus_response.json')

# Parse into DataFrame ready for DB
df_new = _parse_nostradamus_response(resp, project=PROJECT, fm_override=LOCAL_MODEL)
if df_new.empty:
    print('Could not parse forecasts from API response. Inspect /tmp/nostradamus_response.json')
    raise SystemExit(2)

print(f'Parsed {len(df_new)} forecast rows. Inserting into DB...')
con = get_connection()
fm_name = df_new['forecast_method'].iloc[0]
con.execute('DELETE FROM forecasts WHERE project = ? AND forecast_method = ?', [PROJECT, fm_name])
con.register('forecasts_api_df', df_new)
con.execute('INSERT INTO forecasts (project, forecast_method, item_id, forecast_date, forecast) SELECT project, forecast_method, item_id, forecast_date, forecast FROM forecasts_api_df')
con.close()
print(f'Inserted {len(df_new)} rows for project={PROJECT}, method={fm_name}.')
