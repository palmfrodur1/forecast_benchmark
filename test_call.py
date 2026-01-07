import json
import os

import requests


def main() -> None:
    base_url = os.getenv("NOSTRADAMUS_API_BASE_URL", "https://api.nostradamus-api.com")
    url = base_url.rstrip("/") + "/api/v1/forecast/generate"

    # Small sample payload (monthly example)
    payload = {
        "sim_input_his": [
            {"item_id": "001", "actual_sale": 40, "day": "2025-01-01"},
            {"item_id": "001", "actual_sale": 42, "day": "2025-02-01"},
            {"item_id": "001", "actual_sale": 45, "day": "2025-03-01"},
            {"item_id": "001", "actual_sale": 50, "day": "2025-04-01"},
            {"item_id": "001", "actual_sale": 48, "day": "2025-05-01"},
        ],
        "forecast_periods": 6,
        "mode": "local",
        "local_model": "auto_ets",
        "season_length": 12,
        "freq": "MS",
    }

    print("Sending request to", url)
    try:
        headers = {}
        api_key = os.getenv("NOSTRADAMUS_API_KEY")
        if api_key:
            headers["X-API-Key"] = api_key
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        print("Status code:", resp.status_code)
        try:
            print(json.dumps(resp.json(), indent=2))
        except Exception:
            print("Response text:", resp.text)
    except Exception as e:
        print("Request failed:", e)


if __name__ == "__main__":
    main()
