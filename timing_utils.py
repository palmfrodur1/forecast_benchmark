from __future__ import annotations

from datetime import datetime, timezone
import time


def _now_local_iso() -> str:
    # Local time with offset, stable and easy to scan in logs.
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S%z")


def fmt_elapsed(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem_s = seconds - minutes * 60
    if minutes < 60:
        return f"{minutes}m{rem_s:04.1f}s"
    hours = int(minutes // 60)
    rem_m = minutes - hours * 60
    return f"{hours}h{rem_m:02d}m{rem_s:04.1f}s"


def start_timer() -> float:
    return time.perf_counter()


def elapsed_since(start: float) -> float:
    return time.perf_counter() - start


def log(msg: str, *, since: float | None = None) -> None:
    prefix = _now_local_iso()
    if since is not None:
        prefix = f"{prefix} (+{fmt_elapsed(elapsed_since(since))})"
    print(f"[{prefix}] {msg}", flush=True)
