"""
Shared scraping runtime data models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.domain.canonical_insight import CanonicalInsightInput


@dataclass
class ParsedCompetitorData:
    """
    Parsed and grouped competitor data before canonical normalization.
    """

    pricing_items: list[dict[str, Any]] = field(default_factory=list)
    product_items: list[dict[str, Any]] = field(default_factory=list)
    marketing_headlines: list[dict[str, Any]] = field(default_factory=list)
    event_announcements: list[dict[str, Any]] = field(default_factory=list)

    def merge(self, other: "ParsedCompetitorData") -> None:
        self.pricing_items.extend(other.pricing_items)
        self.product_items.extend(other.product_items)
        self.marketing_headlines.extend(other.marketing_headlines)
        self.event_announcements.extend(other.event_announcements)


@dataclass(frozen=True)
class ScraperRunResult:
    """
    Outcome for one scraper execution.
    """

    competitor: str
    records: list[CanonicalInsightInput]
    failed_pages: int
    errors: list[str]
