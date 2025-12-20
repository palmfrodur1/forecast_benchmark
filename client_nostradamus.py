from __future__ import annotations

import hmac
import hashlib
import time
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests


def _validate_base_url(base_url: str) -> None:
    parsed = urlparse(base_url)
    if parsed.scheme == "https" and (parsed.hostname in {"localhost", "127.0.0.1", "0.0.0.0"}):
        raise ValueError(
            "base_url uses https with a localhost address. "
            "Unless your local API is configured for TLS, use http://localhost:<port> instead. "
            f"Got: {base_url}"
        )


def submit_forecast_job(
    base_url: str,
    payload: Dict[str, Any],
    webhook_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_s: float = 30.0,
) -> Dict[str, Any]:
    """Submit an async forecast job.

    Calls:
      POST /api/v1/forecast/generate_job[?webhook_url=...]

    Returns a job envelope such as:
      {"job_id": "...", "status_url": "...", "status": "pending"}

    Notes:
    - If provided, `api_key` is sent as `X-API-Key`.
    """

    url = base_url.rstrip("/") + "/api/v1/forecast/generate_job"
    _validate_base_url(base_url)
    params = {"webhook_url": webhook_url} if webhook_url else None

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    resp = requests.post(url, params=params, json=payload, headers=headers, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def get_job_status(
    base_url: str,
    job_id: str,
    api_key: Optional[str] = None,
    timeout_s: float = 30.0,
) -> Dict[str, Any]:
    """Fetch the job record.

    Calls:
      GET /api/v1/forecast/jobs/{job_id}

    Returns a record with status + result/error fields.
    """

    url = base_url.rstrip("/") + f"/api/v1/forecast/jobs/{job_id}"
    _validate_base_url(base_url)
    headers: Dict[str, str] = {}
    if api_key:
        headers["X-API-Key"] = api_key

    resp = requests.get(url, headers=headers, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def wait_for_job(
    base_url: str,
    job_id: str,
    api_key: Optional[str] = None,
    timeout_total_s: float = 120.0,
    poll_interval_s: float = 0.5,
) -> Dict[str, Any]:
    """Poll until the job finishes or fails (or times out)."""

    deadline = time.time() + timeout_total_s
    while True:
        status_obj = get_job_status(base_url, job_id, api_key=api_key)
        status = status_obj.get("status")
        if status in ("finished", "failed"):
            return status_obj

        if time.time() > deadline:
            raise TimeoutError(
                f"Job {job_id} did not finish in {timeout_total_s}s (last status={status})"
            )

        time.sleep(poll_interval_s)


def compute_webhook_signature(secret: str, timestamp: str, body: bytes) -> str:
    """Compute signature string `sha256=<hex>`.

    Signing scheme:
      sig = HMAC_SHA256(secret, ts + '.' + raw_body_bytes)
    """

    msg = timestamp.encode("utf-8") + b"." + body
    digest = hmac.new(secret.encode("utf-8"), msg=msg, digestmod=hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def verify_webhook_signature(
    *,
    secret: str,
    timestamp: str,
    body: bytes,
    signature_header: str,
) -> bool:
    """Verify `X-Signature` header using constant-time comparison."""

    expected = compute_webhook_signature(secret, timestamp, body)
    # Compare the full `sha256=...` string.
    return hmac.compare_digest(expected, signature_header.strip())
