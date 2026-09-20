"""In-process metrics counters for /metrics."""
from __future__ import annotations

import threading
import time
from collections import Counter

_lock = threading.Lock()
_counters: Counter = Counter()
_started_at = time.time()


def incr(name: str, n: int = 1) -> None:
    with _lock:
        _counters[name] += n


def snapshot() -> dict:
    with _lock:
        data = dict(_counters)
    data["uptime_seconds"] = round(time.time() - _started_at, 1)
    return data
