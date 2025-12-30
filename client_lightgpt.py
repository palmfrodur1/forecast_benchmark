from __future__ import annotations

from typing import Any, Dict, Optional

import requests
from requests import HTTPError


def call_lightgpt_batch(
    base_url: str,
    payload: Dict[str, Any],
    api_key: Optional[str] = None,
    timeout_s: float = 60.0,
) -> Dict[str, Any]:
    """Call Nostradamus LightGPT batch endpoint.

    Endpoint:
      POST /api/v1/lightgpt/batch

    Auth:
      If provided, `api_key` is sent as `X-API-Key`.
    """

    url = base_url.rstrip('/') + '/api/v1/lightgpt/batch'

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    resp = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
    try:
      resp.raise_for_status()
    except HTTPError as e:
      body = None
      try:
        body = resp.text
      except Exception:
        body = None
      msg = f"LightGPT HTTP {resp.status_code} for {resp.url}"
      if body:
        # Avoid exploding logs; keep it reasonably small.
        msg += f"\nResponse body (truncated):\n{body[:4000]}"
      raise RuntimeError(msg) from e
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError("LightGPT response was not a JSON object")
    return data
