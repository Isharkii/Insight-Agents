"""
app/connectors/world_bank_connector.py

World Bank connector for macro-economic indicators.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Mapping

from app.config import ExternalHTTPSettings, WorldBankSettings
from app.connectors.base import BaseConnector, ConnectorFetchResult, ConnectorRequestError
from app.connectors.macro import (
    BaseMacroProvider,
    MacroProviderError,
    WorldBankMacroProvider,
)
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
        provider: BaseMacroProvider | None = None,
    ) -> None:
        super().__init__(source="world_bank", http_settings=http_settings)
        self._settings = settings
        self._provider = provider or WorldBankMacroProvider(
            http_settings=http_settings,
            base_url=settings.base_url,
            per_page=settings.per_page,
            latest_periods=settings.latest_periods,
        )

    def fetch_records(self) -> ConnectorFetchResult:
        if not self._settings.enabled:
            return ConnectorFetchResult(source=self.source, records=[], failed_records=0)

        try:
            rows = self._provider.fetch(
                country=self._settings.country_code,
                metric=self._settings.indicator_code,
                limit=self._settings.latest_periods,
            )
        except MacroProviderError as exc:
            logger.error("World Bank macro provider failed: %s", exc)
            raise ConnectorRequestError(f"{self.source}: {exc}") from exc

        records: list[CanonicalInsightInput] = []
        failed_records = 0
        for index, row in enumerate(rows):
            try:
                normalized = self._normalize_observation(row)
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

    def _normalize_observation(self, row: Mapping[str, Any]) -> CanonicalInsightInput | None:
        if not isinstance(row, Mapping):
            return None

        value = row.get("value")
        period_end = str(row.get("period_end") or "").strip()
        period_start = str(row.get("period_start") or "").strip() or None
        metric_name = str(row.get("metric") or self._settings.indicator_code).strip() or self._settings.indicator_code
        if value is None or not period_end:
            return None

        timestamp = self._parse_period_end(period_end)
        if timestamp is None:
            return None

        country_name = str(row.get("country") or self._settings.country_code).strip()
        source_name = str(row.get("source") or self.source).strip() or self.source

        metadata_json = {
            "country_code": self._settings.country_code,
            "provider": source_name,
            "period_start": period_start,
            "period_end": period_end,
        }

        return CanonicalInsightInput(
            source_type=CanonicalSourceType.API,
            entity_name=country_name,
            category=CanonicalCategory.MACRO,
            metric_name=metric_name,
            metric_value=value,
            timestamp=timestamp,
            region=self._settings.country_code,
            metadata_json=metadata_json,
        )

    @staticmethod
    def _parse_period_end(period_raw: str) -> datetime | None:
        try:
            parsed = datetime.fromisoformat(period_raw.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
