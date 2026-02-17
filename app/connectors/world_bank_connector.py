"""
app/connectors/world_bank_connector.py

World Bank connector for macro-economic indicators.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from app.config import ExternalHTTPSettings, WorldBankSettings
from app.connectors.base import BaseConnector, ConnectorFetchResult
from app.domain.canonical_insight import CanonicalInsightInput
from db.models.canonical_insight_record import CanonicalCategory, CanonicalSourceType

logger = logging.getLogger(__name__)


class WorldBankConnector(BaseConnector):
    """
    Connector for ingesting economic indicator data from the World Bank API.
    """

    def __init__(
        self,
        *,
        settings: WorldBankSettings,
        http_settings: ExternalHTTPSettings,
    ) -> None:
        super().__init__(source="world_bank", http_settings=http_settings)
        self._settings = settings

    def fetch_records(self) -> ConnectorFetchResult:
        if not self._settings.enabled:
            return ConnectorFetchResult(source=self.source, records=[], failed_records=0)

        endpoint = (
            f"{self._settings.base_url.rstrip('/')}/country/"
            f"{self._settings.country_code}/indicator/{self._settings.indicator_code}"
        )
        payload = self._request_json(
            method="GET",
            url=endpoint,
            params={
                "format": "json",
                "per_page": self._settings.per_page,
                "mrv": self._settings.latest_periods,
            },
        )

        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            logger.error("Unexpected World Bank payload shape.")
            return ConnectorFetchResult(source=self.source, records=[], failed_records=1)

        rows = payload[1]
        records: list[CanonicalInsightInput] = []
        failed_records = 0
        for index, row in enumerate(rows):
            try:
                normalized = self._normalize_row(row)
                if normalized is None:
                    failed_records += 1
                    continue
                records.append(normalized)
            except Exception as exc:
                failed_records += 1
                logger.warning(
                    "Failed to normalize World Bank row index=%s error=%s",
                    index,
                    exc,
                )

        return ConnectorFetchResult(
            source=self.source,
            records=records,
            failed_records=failed_records,
        )

    def _normalize_row(self, row: Any) -> CanonicalInsightInput | None:
        if not isinstance(row, dict):
            return None

        value = row.get("value")
        period_raw = (row.get("date") or "").strip()
        if value is None or not period_raw:
            return None

        timestamp = self._parse_world_bank_period(period_raw)
        if timestamp is None:
            return None

        country_meta = row.get("country") if isinstance(row.get("country"), dict) else {}
        indicator_meta = row.get("indicator") if isinstance(row.get("indicator"), dict) else {}
        country_name = (country_meta.get("value") or self._settings.country_code).strip()

        metadata_json = {
            "country_id": country_meta.get("id"),
            "country_code": self._settings.country_code,
            "indicator_name": indicator_meta.get("value"),
            "obs_status": row.get("obs_status"),
            "decimal_places": row.get("decimal"),
        }

        return CanonicalInsightInput(
            source_type=CanonicalSourceType.API,
            entity_name=country_name,
            category=CanonicalCategory.MACRO,
            metric_name=self._settings.indicator_code,
            metric_value=value,
            timestamp=timestamp,
            region=self._settings.country_code,
            metadata_json=metadata_json,
        )

    @staticmethod
    def _parse_world_bank_period(period_raw: str) -> datetime | None:
        try:
            year = int(period_raw[:4])
            return datetime(year=year, month=1, day=1, tzinfo=timezone.utc)
        except (TypeError, ValueError):
            return None
