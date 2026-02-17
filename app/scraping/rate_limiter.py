"""
Domain-aware request rate limiter.
"""

from __future__ import annotations

import threading
import time
from urllib.parse import urlparse


class DomainRateLimiter:
    """
    Enforces minimum interval between requests per domain.
    """

    def __init__(self, *, default_rate_limit_per_second: float) -> None:
        self._default_rate_limit_per_second = max(0.1, default_rate_limit_per_second)
        self._last_request_by_domain: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(
        self,
        *,
        url: str,
        rate_limit_per_second: float | None = None,
        crawl_delay_seconds: float | None = None,
    ) -> None:
        """
        Sleep as needed so outbound requests respect per-domain throttling.
        """

        parsed = urlparse(url)
        domain = parsed.netloc.lower() or parsed.path.lower()
        if not domain:
            return

        effective_rps = max(
            0.1,
            rate_limit_per_second or self._default_rate_limit_per_second,
        )
        min_interval = 1.0 / effective_rps
        if crawl_delay_seconds is not None:
            min_interval = max(min_interval, max(0.0, crawl_delay_seconds))

        with self._lock:
            now = time.monotonic()
            last_time = self._last_request_by_domain.get(domain, 0.0)
            elapsed = now - last_time
            wait_seconds = min_interval - elapsed
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            self._last_request_by_domain[domain] = time.monotonic()
