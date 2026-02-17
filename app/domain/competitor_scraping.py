"""
app/domain/competitor_scraping.py

Domain models for competitor scraping orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CompetitorScrapeSummary:
    """
    Summary for one competitor scrape run.
    """

    competitor: str
    records_scraped: int
    records_inserted: int
    failed_pages: int
    status: str
    errors: list[str] = field(default_factory=list)
