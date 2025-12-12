import requests
import json

url = "https://api.nostradamus.com/api/v1/forecast/generate"

# Small sample payload (monthly example)
payload = {
    "sim_input_his": [
        {"item_id": "001", "actual_sale": 40, "day": "2025-01-01"},
        {"item_id": "001", "actual_sale": 42, "day": "2025-02-01"},
        {"item_id": "001", "actual_sale": 45, "day": "2025-03-01"},
        {"item_id": "001", "actual_sale": 50, "day": "2025-04-01"},
        {"item_id": "001", "actual_sale": 48, "day": "2025-05-01"}
    ],
    "forecast_periods": 6,
    "mode": "local",
    "local_model": "auto_ets",
    "season_length": 12,
    "freq": "MS"
}

print('Sending request to', url)
try:
    resp = requests.post(url, json=payload, timeout=60)
    print('Status code:', resp.status_code)
    try:
        print(json.dumps(resp.json(), indent=2))
    except Exception:
        print('Response text:', resp.text)
except Exception as e:
    print('Request failed:', e)
