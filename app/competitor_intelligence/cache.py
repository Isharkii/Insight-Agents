"""Small async-safe in-memory TTL cache."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass(slots=True)
class _CacheEntry(Generic[T]):
    value: T
    expires_at: float


class AsyncTTLCache(Generic[T]):
    """In-memory cache with expiration and bounded size."""

    def __init__(self, *, ttl_seconds: int, max_size: int) -> None:
        self._ttl_seconds = int(ttl_seconds)
        self._max_size = int(max_size)
        self._store: dict[str, _CacheEntry[T]] = {}
        # RLock keeps cache operations thread-safe across concurrent worker threads.
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> T | None:
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.expires_at <= now:
                self._store.pop(key, None)
                self._misses += 1
                return None
            self._hits += 1
            return entry.value

    async def set(self, key: str, value: T) -> None:
        now = time.monotonic()
        with self._lock:
            if len(self._store) >= self._max_size:
                self._evict_oldest()
            self._store[key] = _CacheEntry(value=value, expires_at=now + self._ttl_seconds)

    async def clear(self) -> None:
        with self._lock:
            self._store.clear()

    async def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._store),
                "hits": self._hits,
                "misses": self._misses,
            }

    def _evict_oldest(self) -> None:
        if not self._store:
            return
        oldest_key = min(self._store.keys(), key=lambda k: self._store[k].expires_at)
        self._store.pop(oldest_key, None)
