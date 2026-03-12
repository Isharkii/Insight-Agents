from __future__ import annotations

import threading
import time
from collections import deque


class SlidingWindowRateLimiter:
    """
    In-process sliding-window limiter keyed by tenant and route.
    """

    def __init__(self, *, max_requests: int, window_seconds: int) -> None:
        self._max_requests = max(1, int(max_requests))
        self._window_seconds = max(1, int(window_seconds))
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, int]:
        now = time.monotonic()
        threshold = now - self._window_seconds
        with self._lock:
            bucket = self._hits.get(key)
            if bucket is None:
                bucket = deque()
                self._hits[key] = bucket
            while bucket and bucket[0] < threshold:
                bucket.popleft()

            if len(bucket) >= self._max_requests:
                retry_after = int(max(1.0, self._window_seconds - (now - bucket[0])))
                return False, retry_after

            bucket.append(now)
            return True, 0
