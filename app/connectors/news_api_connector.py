"""
app/connectors/news_api_connector.py

News API connector for global event ingestion.
"""

from __future__ import annotations

import logging
from datetime import timezone
from typing import Any

from app.config import ExternalHTTPSettings, NewsAPISettings
from app.connectors.base import BaseConnector, ConnectorFetchResult
from app.domain.canonical_insight import CanonicalInsightInput
from db.models.canonical_insight_record import CanonicalCategory, CanonicalSourceType

logger = logging.getLogger(__name__)


class NewsAPIConnector(BaseConnector):
    """
    Connector for ingesting global event articles from NewsAPI.
    """

    def __init__(
        self,
        *,
        settings: NewsAPISettings,
        http_settings: ExternalHTTPSettings,
    ) -> None:
        super().__init__(source="news_api", http_settings=http_settings)
        self._settings = settings

    def fetch_records(self) -> ConnectorFetchResult:
        if not self._settings.enabled:
            return ConnectorFetchResult(source=self.source, records=[], failed_records=0)

        if not self._settings.api_key:
            logger.error("NEWS_API_KEY is missing; skipping News API ingestion.")
            return ConnectorFetchResult(source=self.source, records=[], failed_records=1)

        payload = self._request_json(
            method="GET",
            url=self._settings.base_url,
            params={
                "q": self._settings.query,
                "language": self._settings.language,
                "pageSize": self._settings.page_size,
                "sortBy": "publishedAt",
            },
            headers={"X-Api-Key": self._settings.api_key},
        )

        articles = payload.get("articles", []) if isinstance(payload, dict) else []
        records: list[CanonicalInsightInput] = []
        failed_records = 0

        for index, article in enumerate(articles):
            try:
                parsed = self._normalize_article(article)
                if parsed is None:
                    failed_records += 1
                    continue
                records.append(parsed)
            except Exception as exc:
                failed_records += 1
                logger.warning(
                    "Failed to normalize News API record index=%s error=%s",
                    index,
                    exc,
                )

        return ConnectorFetchResult(
            source=self.source,
            records=records,
            failed_records=failed_records,
        )

    def _normalize_article(self, article: Any) -> CanonicalInsightInput | None:
        if not isinstance(article, dict):
            return None

        title = (article.get("title") or "").strip()
        published_raw = (article.get("publishedAt") or "").strip()
        if not title or not published_raw:
            return None

        try:
            timestamp = self.parse_iso_datetime(published_raw)
        except ValueError:
            return None

        source_meta = article.get("source") if isinstance(article.get("source"), dict) else {}
        entity_name = (source_meta.get("name") or "global_news").strip()
        if not entity_name:
            entity_name = "global_news"

        metric_payload = {
            "title": title,
            "description": article.get("description"),
            "url": article.get("url"),
        }

        metadata_json = {
            "author": article.get("author"),
            "query": self._settings.query,
            "language": self._settings.language,
            "content_preview": article.get("content"),
        }

        return CanonicalInsightInput(
            source_type=CanonicalSourceType.API,
            entity_name=entity_name,
            category=CanonicalCategory.EVENT,
            metric_name="news_article",
            metric_value=metric_payload,
            timestamp=timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc),
            region=self._settings.region,
            metadata_json=metadata_json,
        )
