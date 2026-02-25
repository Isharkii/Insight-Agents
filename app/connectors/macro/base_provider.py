"""
app/connectors/macro/base_provider.py

Macro API provider abstraction with normalized outputs and shared HTTP behavior.
"""

from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Any, TypedDict

import requests

from app.config import ExternalHTTPSettings

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {500, 502, 503, 504}


class MacroObservation(TypedDict):
    """
    Normalized macro observation shared by all providers.
    """

    country: str
    metric: str
    period_start: str
    period_end: str
    value: float
    source: str


class MacroProviderError(RuntimeError):
    """
    Base class for macro provider failures.
    """


class MacroProviderConfigurationError(MacroProviderError):
    """
    Raised when a provider is missing mandatory configuration.
    """


class MacroProviderUnsupportedError(MacroProviderError):
    """
    Raised when a provider is not available or not implemented.
    """


class MacroProviderRequestError(MacroProviderError):
    """
    Raised when HTTP requests fail after retries.
    """


class MacroProviderRateLimitError(MacroProviderRequestError):
    """
    Raised when provider rate limits are exceeded after retries.
    """


class MacroProviderResponseError(MacroProviderError):
    """
    Raised when provider payloads are malformed or unexpected.
    """


class BaseMacroProvider(ABC):
    """
    Provider interface for fetching normalized macro observations.
    """

    provider_name: str

    def __init__(
        self,
        *,
        source: str,
        http_settings: ExternalHTTPSettings,
        session: requests.Session | None = None,
    ) -> None:
        self.source = source
        self._session = session or requests.Session()
        self._timeout_seconds = max(0.1, float(http_settings.timeout_seconds))
        self._max_retries = max(0, int(http_settings.max_retries))
        self._backoff_initial_seconds = max(0.0, float(http_settings.backoff_initial_seconds))
        self._backoff_multiplier = max(1.0, float(http_settings.backoff_multiplier))
        rate_limit = float(http_settings.rate_limit_per_second)
        self._min_request_interval_seconds = 1.0 / rate_limit if rate_limit > 0 else 0.0
        self._last_request_monotonic = 0.0

    @abstractmethod
    def fetch(
        self,
        *,
        country: str,
        metric: str,
        period_start: str | None = None,
        period_end: str | None = None,
        limit: int | None = None,
    ) -> list[MacroObservation]:
        """
        Fetch macro observations normalized to the shared response structure.
        """

    def _request_json(
        self,
        *,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        response = self._request(method=method, url=url, params=params, headers=headers)
        try:
            return response.json()
        except ValueError as exc:
            raise MacroProviderResponseError(f"{self.source}: response is not valid JSON.") from exc

    def _request(
        self,
        *,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> requests.Response:
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
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    break
                self._sleep_backoff(attempt)
                continue

            status_code = int(response.status_code)
            if status_code == 429:
                if attempt >= self._max_retries:
                    raise MacroProviderRateLimitError(
                        f"{self.source}: rate limit exceeded after retries."
                    )
                retry_after = self._parse_retry_after_seconds(response.headers.get("Retry-After"))
                wait_seconds = retry_after if retry_after is not None else self._backoff_seconds(attempt)
                logger.warning(
                    "Macro provider rate limited source=%s attempt=%s/%s wait_seconds=%.2f url=%s",
                    self.source,
                    attempt + 1,
                    self._max_retries,
                    wait_seconds,
                    url,
                )
                time.sleep(wait_seconds)
                continue

            if status_code in RETRYABLE_STATUS_CODES:
                last_error = MacroProviderRequestError(
                    f"{self.source}: retryable HTTP status {status_code}."
                )
                if attempt >= self._max_retries:
                    break
                self._sleep_backoff(attempt)
                continue

            if status_code >= 400:
                detail = (response.text or "").strip()
                if len(detail) > 240:
                    detail = detail[:240] + "..."
                raise MacroProviderRequestError(
                    f"{self.source}: non-retryable HTTP {status_code}. {detail}".strip()
                )

            return response

        raise MacroProviderRequestError(f"{self.source}: request failed after retries.") from last_error

    def _build_observation(
        self,
        *,
        country: str,
        metric: str,
        period_start: Any,
        period_end: Any,
        value: Any,
    ) -> MacroObservation | None:
        parsed_value = self._coerce_float(value)
        if parsed_value is None:
            return None

        normalized_period_start = self.normalize_period(period_start)
        normalized_period_end = self.normalize_period(period_end)
        if normalized_period_start is None or normalized_period_end is None:
            return None

        normalized_country = str(country or "").strip().upper()
        normalized_metric = str(metric or "").strip()
        if not normalized_country or not normalized_metric:
            return None

        return {
            "country": normalized_country,
            "metric": normalized_metric,
            "period_start": normalized_period_start,
            "period_end": normalized_period_end,
            "value": parsed_value,
            "source": self.source,
        }

    def _apply_rate_limit(self) -> None:
        if self._min_request_interval_seconds <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._last_request_monotonic
        remaining = self._min_request_interval_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_monotonic = time.monotonic()

    def _sleep_backoff(self, attempt: int) -> None:
        time.sleep(self._backoff_seconds(attempt))

    def _backoff_seconds(self, attempt: int) -> float:
        return self._backoff_initial_seconds * (self._backoff_multiplier**attempt)

    @staticmethod
    def normalize_period(value: Any) -> str | None:
        if isinstance(value, datetime):
            return value.date().isoformat()
        if isinstance(value, date):
            return value.isoformat()
        if not isinstance(value, str):
            return None

        raw = value.strip()
        if not raw:
            return None
        raw = raw.replace("Z", "+00:00")

        # YYYY-MM-DD
        try:
            return date.fromisoformat(raw[:10]).isoformat()
        except ValueError:
            pass

        # Full datetime input.
        try:
            return datetime.fromisoformat(raw).date().isoformat()
        except ValueError:
            return None

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(parsed):
            return None
        return float(parsed)

    @staticmethod
    def _parse_retry_after_seconds(value: Any) -> float | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = float(text)
        except ValueError:
            return None
        return max(0.0, parsed)
