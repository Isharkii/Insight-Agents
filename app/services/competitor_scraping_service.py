"""
app/services/competitor_scraping_service.py

Service orchestration for competitor website scraping.
"""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy.orm import Session

from app.domain.competitor_scraping import CompetitorScrapeSummary
from app.scraping.config import get_competitor_scraping_settings
from app.scraping.engine import CompetitorScrapingEngine
from app.scraping.storage import SQLAlchemyInsightStorage


class CompetitorScrapingService:
    """
    Runs competitor scraping pipeline and persists canonical records.
    """

    def __init__(self) -> None:
        self._settings = get_competitor_scraping_settings()

    def ingest(
        self,
        *,
        db: Session,
        competitor: str | None = None,
    ) -> list[CompetitorScrapeSummary]:
        storage = SQLAlchemyInsightStorage(
            session=db,
            batch_size=self._settings.storage_batch_size,
        )
        engine = CompetitorScrapingEngine(
            settings=self._settings,
            storage=storage,
        )
        selected = [competitor] if competitor else None
        return engine.run(competitors=selected)


@lru_cache(maxsize=1)
def get_competitor_scraping_service() -> CompetitorScrapingService:
    """
    Build and cache competitor scraping service.
    """

    return CompetitorScrapingService()
