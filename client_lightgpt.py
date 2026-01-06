from __future__ import annotations

from typing import Any, Dict, Optional

import requests
from requests import HTTPError


def _normalize_lightgpt_freq(freq: Optional[str]) -> str:
  """Normalize frequency values for LightGPT.

  The LightGPT API expects monthly as 'M' (and may reject 'MS').
  We treat missing/unknown monthly-ish values as 'M' by default.
  """

  if freq is None:
    return 'M'
  f = str(freq).strip().lower()
  if f in {'', 'none'}:
    return 'M'
  if f in {'month', 'monthly', 'm', 'ms'}:
    return 'M'
  if f in {'d', 'day', 'daily'}:
    return 'D'
  return str(freq)


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

    # Default LightGPT to monthly frequency unless explicitly overridden.
    payload_to_send = dict(payload or {})
    payload_to_send['freq'] = _normalize_lightgpt_freq(payload_to_send.get('freq'))

    url = base_url.rstrip('/') + '/api/v1/lightgpt/batch'

    headers = {"Content-Type": "application/json"}
    if api_key:
      headers["X-API-Key"] = api_key

    # The LightGPT OpenAPI schema uses `api_key` for the Nixtla key.
    # As a fallback for some deployments, also send it in a dedicated header.
    # Never log or print the key value.
    nixtla_key = payload_to_send.get('api_key')
    if isinstance(nixtla_key, str) and nixtla_key.strip():
      headers['X-NIXTLA-API-KEY'] = nixtla_key.strip()

    resp = requests.post(url, json=payload_to_send, headers=headers, timeout=timeout_s)
    try:
      resp.raise_for_status()
    except HTTPError as e:
      body = None
      try:
        body = resp.text
      except Exception:
        body = None
      sent_freq = payload_to_send.get('freq')
      has_nixtla = bool(isinstance(payload_to_send.get('api_key'), str) and payload_to_send.get('api_key').strip())
      msg = (
        f"LightGPT HTTP {resp.status_code} for {resp.url} "
        f"(sent freq={sent_freq!r}, sent_has_api_key={has_nixtla})"
      )
      if body:
        # Avoid exploding logs; keep it reasonably small.
        msg += f"\nResponse body (truncated):\n{body[:4000]}"
      raise RuntimeError(msg) from e
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError("LightGPT response was not a JSON object")
    return data
