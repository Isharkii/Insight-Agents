"""
Config-driven scraper implementation.
"""

from __future__ import annotations

from bs4 import BeautifulSoup

from app.scraping.base import ScraperBase
from app.scraping.parsing import HTMLParsingLayer
from app.scraping.types import ParsedCompetitorData


class ConfigurableCompetitorScraper(ScraperBase):
    """
    Scraper that relies on page selectors from competitor config.
    """

    def parse_page(
        self,
        *,
        page_kind: str,
        page_url: str,
        soup: BeautifulSoup,
    ) -> ParsedCompetitorData:
        return HTMLParsingLayer.parse_page(
            page_kind=page_kind,
            page_url=page_url,
            soup=soup,
            selectors=self.selectors_for(page_kind),
        )

    def selectors_for(self, page_kind: str) -> list[str]:
        return self.config.selectors.get(page_kind.strip().lower(), [])
