"""
app/connectors/macro/fred_provider.py

FRED macro provider implementation.
"""

from __future__ import annotations

import requests

from app.config import ExternalHTTPSettings
from app.connectors.macro.base_provider import (
    BaseMacroProvider,
    MacroObservation,
    MacroProviderConfigurationError,
    MacroProviderResponseError,
)


class FREDMacroProvider(BaseMacroProvider):
    """
    Macro API provider for Federal Reserve Economic Data (FRED).
    """

    provider_name = "fred"

    def __init__(
        self,
        *,
        http_settings: ExternalHTTPSettings,
        api_key: str | None = None,
        base_url: str = "https://api.stlouisfed.org/fred/series/observations",
        default_limit: int = 240,
        session: requests.Session | None = None,
    ) -> None:
        super().__init__(source=self.provider_name, http_settings=http_settings, session=session)
        self._api_key = str(api_key or "").strip() or None
        self._base_url = str(base_url or "https://api.stlouisfed.org/fred/series/observations").strip()
        self._default_limit = max(1, int(default_limit))

    def fetch(
        self,
        *,
        country: str,
        metric: str,
        period_start: str | None = None,
        period_end: str | None = None,
        limit: int | None = None,
    ) -> list[MacroObservation]:
        if not self._api_key:
            raise MacroProviderConfigurationError("fred: FRED API key is required.")

        normalized_country = str(country or "").strip() or "US"
        normalized_metric = str(metric or "").strip()
        if not normalized_metric:
            return []

        params: dict[str, str | int] = {
            "series_id": normalized_metric,
            "api_key": self._api_key,
            "file_type": "json",
            "sort_order": "asc",
            "limit": max(1, int(limit or self._default_limit)),
        }

        normalized_period_start = self.normalize_period(period_start)
        normalized_period_end = self.normalize_period(period_end)
        if normalized_period_start:
            params["observation_start"] = normalized_period_start
        if normalized_period_end:
            params["observation_end"] = normalized_period_end

        payload = self._request_json(method="GET", url=self._base_url, params=params)
        rows = payload.get("observations") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise MacroProviderResponseError("fred: unexpected payload shape.")

        records: list[MacroObservation] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            date_value = row.get("date")
            value = row.get("value")
            if value in (None, "", "."):
                continue
            observation = self._build_observation(
                country=normalized_country,
                metric=normalized_metric,
                period_start=date_value,
                period_end=date_value,
                value=value,
            )
            if observation is None:
                continue
            records.append(observation)

        records.sort(key=lambda item: (item["period_end"], item["metric"], item["country"]))
        return records
