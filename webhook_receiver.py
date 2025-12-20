from __future__ import annotations

import os
from fastapi import FastAPI, Header, HTTPException, Request

from client_nostradamus import verify_webhook_signature

app = FastAPI(title="Nostradamus Webhook Receiver")


@app.post("/webhook/nostradamus")
async def nostradamus_webhook(
    request: Request,
    x_signature_timestamp: str | None = Header(default=None, alias="X-Signature-Timestamp"),
    x_signature: str | None = Header(default=None, alias="X-Signature"),
):
    """Receives webhooks and verifies the HMAC signature.

    Expected headers:
      - X-Signature-Timestamp: <ts>
      - X-Signature: sha256=<hex>

    The signing scheme is:
      sig = HMAC_SHA256(secret, ts + "." + raw_body_bytes)
    """

    secret = os.getenv("NOSTRADAMUS_WEBHOOK_SECRET")
    if not secret:
        raise HTTPException(status_code=500, detail="NOSTRADAMUS_WEBHOOK_SECRET not configured")

    if not x_signature_timestamp or not x_signature:
        raise HTTPException(status_code=400, detail="Missing signature headers")

    body = await request.body()
    if not verify_webhook_signature(
        secret=secret,
        timestamp=x_signature_timestamp,
        body=body,
        signature_header=x_signature,
    ):
        raise HTTPException(status_code=401, detail="Invalid signature")

    # At this point the webhook is verified.
    # Parse JSON here if your provider sends JSON payloads.
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8001")))
