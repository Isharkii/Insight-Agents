"""
app/connectors/google_trends_connector.py

Google Trends connector using the public trending RSS feed.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any

from app.config import ExternalHTTPSettings, GoogleTrendsSettings
from app.connectors.base import BaseConnector, ConnectorFetchResult
from app.domain.canonical_insight import CanonicalInsightInput
from db.models.canonical_insight_record import CanonicalCategory, CanonicalSourceType

logger = logging.getLogger(__name__)


class GoogleTrendsConnector(BaseConnector):
    """
    Connector for ingesting trending keywords from Google Trends RSS.
    """

    def __init__(
        self,
        *,
        settings: GoogleTrendsSettings,
        http_settings: ExternalHTTPSettings,
    ) -> None:
        super().__init__(source="google_trends", http_settings=http_settings)
        self._settings = settings

    def fetch_records(self) -> ConnectorFetchResult:
        if not self._settings.enabled:
            return ConnectorFetchResult(source=self.source, records=[], failed_records=0)

        xml_text = self._request_text(
            method="GET",
            url=self._settings.rss_url,
            params={"geo": self._settings.geo, "hl": self._settings.hl},
        )

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            logger.error("Failed to parse Google Trends RSS response: %s", exc)
            return ConnectorFetchResult(source=self.source, records=[], failed_records=1)

        records: list[CanonicalInsightInput] = []
        failed_records = 0

        items = root.findall(".//item")
        for index, item in enumerate(items[: self._settings.max_items]):
            try:
                normalized = self._normalize_item(item)
                if normalized is None:
                    failed_records += 1
                    continue
                records.append(normalized)
            except Exception as exc:
                failed_records += 1
                logger.warning(
                    "Failed to normalize Google Trends item index=%s error=%s",
                    index,
                    exc,
                )

        return ConnectorFetchResult(
            source=self.source,
            records=records,
            failed_records=failed_records,
        )

    def _normalize_item(self, item: ET.Element) -> CanonicalInsightInput | None:
        keyword = (self._child_text(item, "title") or "").strip()
        pub_date = (self._child_text(item, "pubDate") or "").strip()
        if not keyword or not pub_date:
            return None

        try:
            timestamp = parsedate_to_datetime(pub_date)
        except (TypeError, ValueError):
            return None
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        approx_traffic_raw = (self._child_text(item, "approx_traffic") or "").strip()
        approx_traffic = self._parse_traffic(approx_traffic_raw)
        link = (self._child_text(item, "link") or "").strip()

        metadata_json: dict[str, Any] = {
            "geo": self._settings.geo,
            "hl": self._settings.hl,
            "link": link or None,
        }
        if approx_traffic_raw:
            metadata_json["approx_traffic_raw"] = approx_traffic_raw

        return CanonicalInsightInput(
            source_type=CanonicalSourceType.SCRAPE,
            entity_name=keyword,
            category=CanonicalCategory.MARKETING,
            metric_name="trend_keyword_traffic",
            metric_value=approx_traffic if approx_traffic is not None else approx_traffic_raw,
            timestamp=timestamp,
            region=self._settings.region,
            metadata_json=metadata_json,
        )

    @staticmethod
    def _child_text(node: ET.Element, local_name: str) -> str | None:
        for child in list(node):
            tag_name = child.tag.rsplit("}", 1)[-1]
            if tag_name == local_name:
                return child.text
        return None

    @staticmethod
    def _parse_traffic(raw_value: str) -> int | None:
        if not raw_value:
            return None

        cleaned = raw_value.strip().upper().replace("+", "").replace(",", "")
        multiplier = 1
        if cleaned.endswith("K"):
            multiplier = 1_000
            cleaned = cleaned[:-1]
        elif cleaned.endswith("M"):
            multiplier = 1_000_000
            cleaned = cleaned[:-1]

        try:
            return int(float(cleaned) * multiplier)
        except ValueError:
            return None
