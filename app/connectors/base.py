"""
app/connectors/base.py

Base connector abstraction and shared HTTP mechanics.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests

from app.config import ExternalHTTPSettings
from app.domain.canonical_insight import CanonicalInsightInput

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class ConnectorRequestError(RuntimeError):
    """
    Raised when a connector cannot fetch data after retries.
    """


@dataclass(frozen=True)
class ConnectorFetchResult:
    """
    Connector fetch outcome with normalized records.
    """

    source: str
    records: list[CanonicalInsightInput]
    failed_records: int = 0


class BaseConnector(ABC):
    """
    Connector interface for fetching and normalizing external records.
    """

    source: str

    def __init__(
        self,
        *,
        source: str,
        http_settings: ExternalHTTPSettings,
        session: requests.Session | None = None,
    ) -> None:
        self.source = source
        self._session = session or requests.Session()
        self._timeout_seconds = http_settings.timeout_seconds
        self._max_retries = http_settings.max_retries
        self._backoff_initial_seconds = http_settings.backoff_initial_seconds
        self._backoff_multiplier = http_settings.backoff_multiplier
        self._min_request_interval_seconds = (
            1.0 / http_settings.rate_limit_per_second if http_settings.rate_limit_per_second > 0 else 0.0
        )
        self._last_request_monotonic: float = 0.0

    @abstractmethod
    def fetch_records(self) -> ConnectorFetchResult:
        """
        Fetch external data and return normalized canonical records.
        """

    def _request_json(
        self,
        *,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """
        Execute an HTTP request and return parsed JSON with retry support.
        """

        response = self._request(method=method, url=url, params=params, headers=headers)
        try:
            return response.json()
        except ValueError as exc:
            raise ConnectorRequestError(f"{self.source}: response was not valid JSON.") from exc

    def _request_text(
        self,
        *,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        """
        Execute an HTTP request and return response text with retry support.
        """

        response = self._request(method=method, url=url, params=params, headers=headers)
        return response.text

    def _request(
        self,
        *,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> requests.Response:
        """
        Execute an HTTP request with rate limiting and exponential backoff.
        """

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            self._apply_rate_limit()
            try:
                response = self._session.request(
                    method=method,
                    url=url,
                    params=params,
                    headers=headers,
                    timeout=self._timeout_seconds,
                )
                if response.status_code in RETRYABLE_STATUS_CODES:
                    raise requests.HTTPError(
                        f"Retryable HTTP status code: {response.status_code}",
                        response=response,
                    )
                response.raise_for_status()
                return response
            except requests.HTTPError as exc:
                last_error = exc
                status_code = exc.response.status_code if exc.response is not None else None
                is_retryable = status_code in RETRYABLE_STATUS_CODES
                if not is_retryable:
                    logger.error(
                        "Connector request failed source=%s status=%s url=%s error=%s",
                        self.source,
                        status_code,
                        url,
                        exc,
                    )
                    raise ConnectorRequestError(f"{self.source}: non-retryable request failure.") from exc
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc

            if attempt >= self._max_retries:
                break

            backoff_seconds = self._backoff_initial_seconds * (self._backoff_multiplier**attempt)
            logger.warning(
                "Connector request retry source=%s attempt=%s/%s wait_seconds=%.2f url=%s",
                self.source,
                attempt + 1,
                self._max_retries,
                backoff_seconds,
                url,
            )
            time.sleep(backoff_seconds)

        logger.error(
            "Connector request exhausted retries source=%s url=%s error=%s",
            self.source,
            url,
            last_error,
        )
        raise ConnectorRequestError(f"{self.source}: request failed after retries.") from last_error

    def _apply_rate_limit(self) -> None:
        """
        Enforce minimum interval between outbound requests.
        """

        if self._min_request_interval_seconds <= 0:
            return

        now = time.monotonic()
        elapsed = now - self._last_request_monotonic
        remaining = self._min_request_interval_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_monotonic = time.monotonic()

    @staticmethod
    def parse_iso_datetime(value: str) -> datetime:
        """
        Parse an ISO datetime string into a timezone-aware datetime.
        """

        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
