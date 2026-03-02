"""Brave Search API implementation of the SearchProvider interface."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests

from app.competitor_intelligence.search.interfaces import (
    SearchProvider,
    SearchResponsePayload,
    SearchResultItem,
)

logger = logging.getLogger(__name__)


class SearchProviderError(RuntimeError):
    """Raised when search retrieval fails."""


class SearchRateLimitError(SearchProviderError):
    """Raised when provider rate limiting persists after retries."""


class BraveSearchProvider(SearchProvider):
    """Search-only provider backed by Brave Web Search API."""

    DEFAULT_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
    DEFAULT_TIMEOUT_SECONDS = 12.0
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_BACKOFF_SECONDS = 1.0
    API_KEY_ENV_VAR = "BRAVE_SEARCH_API_KEY"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
        session: requests.Session | None = None,
    ) -> None:
        resolved_api_key = (api_key or os.getenv(self.API_KEY_ENV_VAR, "")).strip()
        if not resolved_api_key:
            raise ValueError(
                f"Missing Brave API key. Set {self.API_KEY_ENV_VAR} or pass api_key explicitly."
            )
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0.")
        if backoff_seconds <= 0:
            raise ValueError("backoff_seconds must be > 0.")

        self._api_key = resolved_api_key
        self._endpoint = endpoint
        self._timeout_seconds = float(timeout_seconds)
        self._max_retries = int(max_retries)
        self._backoff_seconds = float(backoff_seconds)
        self._session = session or requests.Session()

    def search(self, query: str, *, limit: int = 10) -> SearchResponsePayload:
        query = str(query or "").strip()
        if len(query) < 2:
            raise ValueError("query must be at least 2 characters.")
        if limit < 1 or limit > 50:
            raise ValueError("limit must be between 1 and 50.")

        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self._api_key,
        }
        params = {
            "q": query,
            "count": limit,
        }

        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._session.get(
                    self._endpoint,
                    headers=headers,
                    params=params,
                    timeout=self._timeout_seconds,
                )
                if response.status_code == 429:
                    wait_seconds = self._retry_delay_seconds(
                        attempt=attempt,
                        retry_after=response.headers.get("Retry-After"),
                    )
                    if attempt >= self._max_retries:
                        raise SearchRateLimitError(
                            "Brave API rate limit persisted after retries."
                        )
                    logger.warning(
                        "Brave API rate limited request (attempt %d/%d). Retrying in %.2fs.",
                        attempt,
                        self._max_retries,
                        wait_seconds,
                    )
                    time.sleep(wait_seconds)
                    continue

                if response.status_code >= 500:
                    if attempt >= self._max_retries:
                        raise SearchProviderError(
                            f"Brave API server error {response.status_code} after retries."
                        )
                    wait_seconds = self._retry_delay_seconds(attempt=attempt, retry_after=None)
                    logger.warning(
                        "Brave API server error %d (attempt %d/%d). Retrying in %.2fs.",
                        response.status_code,
                        attempt,
                        self._max_retries,
                        wait_seconds,
                    )
                    time.sleep(wait_seconds)
                    continue

                if response.status_code >= 400:
                    raise SearchProviderError(
                        f"Brave API returned client error {response.status_code}: {response.text[:300]}"
                    )

                payload = response.json()
                return self._normalize_payload(query=query, payload=payload, limit=limit)
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    break
                wait_seconds = self._retry_delay_seconds(attempt=attempt, retry_after=None)
                logger.warning(
                    "Brave request failed (attempt %d/%d): %s. Retrying in %.2fs.",
                    attempt,
                    self._max_retries,
                    exc,
                    wait_seconds,
                )
                time.sleep(wait_seconds)

        raise SearchProviderError(
            f"Failed to retrieve Brave search results for query '{query}' after {self._max_retries} retries: {last_error}"
        )

    def _normalize_payload(
        self,
        *,
        query: str,
        payload: dict[str, Any],
        limit: int,
    ) -> SearchResponsePayload:
        web = payload.get("web")
        raw_results = web.get("results", []) if isinstance(web, dict) else []

        results: list[SearchResultItem] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            results.append(
                {
                    "title": str(item.get("title") or "").strip(),
                    "url": url,
                    "description": str(item.get("description") or "").strip(),
                }
            )
            if len(results) >= limit:
                break

        return {
            "query": query,
            "results": results,
        }

    def _retry_delay_seconds(self, *, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                parsed = float(retry_after)
                if parsed > 0:
                    return parsed
            except ValueError:
                pass
        return self._backoff_seconds * (2 ** max(0, attempt - 1))
