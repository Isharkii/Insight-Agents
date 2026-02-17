"""
SaaS-biased scraper with default selectors for common site patterns.
"""

from __future__ import annotations

from app.scraping.scrapers.configurable_scraper import ConfigurableCompetitorScraper


class SaaSCompetitorScraper(ConfigurableCompetitorScraper):
    """
    Domain-specific scraper subclass for common SaaS website structures.
    """

    DEFAULT_SELECTORS: dict[str, list[str]] = {
        "pricing": [
            ".pricing-card",
            ".plan-card",
            ".pricing-table [class*='plan']",
            "[data-pricing-plan]",
        ],
        "products": [
            ".product-card",
            ".feature-card",
            ".platform-module",
            ".solutions-grid article",
        ],
        "marketing": [
            ".hero h1",
            ".hero h2",
            ".value-prop",
            ".banner-title",
        ],
        "events": [
            ".newsroom article",
            ".press-release",
            ".event-card",
            ".blog-post",
        ],
    }

    def selectors_for(self, page_kind: str) -> list[str]:
        normalized_kind = page_kind.strip().lower()
        configured = super().selectors_for(normalized_kind)
        fallback = self.DEFAULT_SELECTORS.get(normalized_kind, [])
        return [*configured, *fallback]
