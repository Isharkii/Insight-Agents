"""
app/connectors/macro/world_bank_provider.py

World Bank macro provider implementation.
"""

from __future__ import annotations

from typing import Any

import requests

from app.config import ExternalHTTPSettings
from app.connectors.macro.base_provider import (
    BaseMacroProvider,
    MacroObservation,
    MacroProviderResponseError,
)


class WorldBankMacroProvider(BaseMacroProvider):
    """
    Macro API provider for World Bank indicator endpoints.
    """

    provider_name = "world_bank"

    def __init__(
        self,
        *,
        http_settings: ExternalHTTPSettings,
        base_url: str = "https://api.worldbank.org/v2",
        per_page: int = 200,
        latest_periods: int = 20,
        session: requests.Session | None = None,
    ) -> None:
        super().__init__(source=self.provider_name, http_settings=http_settings, session=session)
        self._base_url = str(base_url or "https://api.worldbank.org/v2").rstrip("/")
        self._per_page = max(1, int(per_page))
        self._latest_periods = max(1, int(latest_periods))

    def fetch(
        self,
        *,
        country: str,
        metric: str,
        period_start: str | None = None,
        period_end: str | None = None,
        limit: int | None = None,
    ) -> list[MacroObservation]:
        normalized_country = str(country or "").strip() or "WLD"
        normalized_metric = str(metric or "").strip()
        if not normalized_metric:
            return []

        endpoint = f"{self._base_url}/country/{normalized_country}/indicator/{normalized_metric}"
        params: dict[str, Any] = {
            "format": "json",
            "per_page": self._per_page,
        }

        requested_limit = max(1, int(limit or self._latest_periods))
        if period_start or period_end:
            start_year = self._extract_year(period_start)
            end_year = self._extract_year(period_end)
            if start_year and end_year:
                params["date"] = f"{start_year}:{end_year}"
            elif start_year:
                params["date"] = f"{start_year}:{start_year}"
            elif end_year:
                params["date"] = f"{end_year}:{end_year}"
            params["per_page"] = max(self._per_page, requested_limit)
        else:
            params["mrv"] = requested_limit
            params["per_page"] = max(self._per_page, requested_limit)

        payload = self._request_json(method="GET", url=endpoint, params=params)
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            raise MacroProviderResponseError("world_bank: unexpected payload shape.")

        records: list[MacroObservation] = []
        for row in payload[1]:
            if not isinstance(row, dict):
                continue
            period_bounds = self._period_bounds(row.get("date"))
            if period_bounds is None:
                continue
            observation = self._build_observation(
                country=normalized_country,
                metric=normalized_metric,
                period_start=period_bounds[0],
                period_end=period_bounds[1],
                value=row.get("value"),
            )
            if observation is None:
                continue
            records.append(observation)

        records.sort(key=lambda item: (item["period_end"], item["metric"], item["country"]))
        if limit is not None and len(records) > requested_limit:
            return records[-requested_limit:]
        return records

    @staticmethod
    def _extract_year(value: str | None) -> int | None:
        if value is None:
            return None
        text = str(value).strip()
        if len(text) < 4 or not text[:4].isdigit():
            return None
        parsed = int(text[:4])
        if parsed < 1:
            return None
        return parsed

    @classmethod
    def _period_bounds(cls, value: Any) -> tuple[str, str] | None:
        year = cls._extract_year(str(value) if value is not None else None)
        if year is None:
            return None
        return (f"{year:04d}-01-01", f"{year:04d}-12-31")
