"""
Normalization layer for scraping outputs.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.domain.canonical_insight import CanonicalInsightInput
from app.scraping.config.models import CompetitorDomainConfig
from app.scraping.types import ParsedCompetitorData
from db.models.canonical_insight_record import CanonicalCategory, CanonicalSourceType


class CanonicalNormalizer:
    """
    Convert parsed scrape outputs into canonical persistence records.
    """

    def normalize(
        self,
        *,
        competitor: CompetitorDomainConfig,
        parsed: ParsedCompetitorData,
        scraped_at: datetime | None = None,
    ) -> list[CanonicalInsightInput]:
        normalized_time = scraped_at or datetime.now(timezone.utc)
        if normalized_time.tzinfo is None:
            normalized_time = normalized_time.replace(tzinfo=timezone.utc)

        records: list[CanonicalInsightInput] = []

        for item in parsed.pricing_items:
            records.append(
                CanonicalInsightInput(
                    source_type=CanonicalSourceType.SCRAPE,
                    entity_name=competitor.entity_name,
                    category=CanonicalCategory.PRICING,
                    metric_name="pricing_data",
                    metric_value=item,
                    timestamp=normalized_time,
                    region=competitor.region,
                    metadata_json={
                        "competitor": competitor.name,
                        "page_url": item.get("page_url"),
                        "record_type": "pricing",
                    },
                )
            )

        for item in parsed.product_items:
            records.append(
                CanonicalInsightInput(
                    source_type=CanonicalSourceType.SCRAPE,
                    entity_name=competitor.entity_name,
                    category=CanonicalCategory.SALES,
                    metric_name="product_listing",
                    metric_value=item,
                    timestamp=normalized_time,
                    region=competitor.region,
                    metadata_json={
                        "competitor": competitor.name,
                        "page_url": item.get("page_url"),
                        "record_type": "product",
                    },
                )
            )

        for item in parsed.marketing_headlines:
            records.append(
                CanonicalInsightInput(
                    source_type=CanonicalSourceType.SCRAPE,
                    entity_name=competitor.entity_name,
                    category=CanonicalCategory.MARKETING,
                    metric_name="marketing_headline",
                    metric_value=item.get("headline"),
                    timestamp=normalized_time,
                    region=competitor.region,
                    metadata_json={
                        "competitor": competitor.name,
                        "page_url": item.get("page_url"),
                        "record_type": "headline",
                    },
                )
            )

        for item in parsed.event_announcements:
            event_timestamp = self._event_timestamp(item.get("event_date"), normalized_time)
            records.append(
                CanonicalInsightInput(
                    source_type=CanonicalSourceType.SCRAPE,
                    entity_name=competitor.entity_name,
                    category=CanonicalCategory.EVENT,
                    metric_name="event_announcement",
                    metric_value=item,
                    timestamp=event_timestamp,
                    region=competitor.region,
                    metadata_json={
                        "competitor": competitor.name,
                        "page_url": item.get("page_url"),
                        "record_type": "event",
                    },
                )
            )

        return records

    @staticmethod
    def _event_timestamp(value: object, fallback: datetime) -> datetime:
        if isinstance(value, str) and value:
            normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
            try:
                parsed = datetime.fromisoformat(normalized)
                if parsed.tzinfo is None:
                    return parsed.replace(tzinfo=timezone.utc)
                return parsed
            except ValueError:
                return fallback
        return fallback
