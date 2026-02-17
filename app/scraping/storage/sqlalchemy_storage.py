"""
SQLAlchemy-backed storage implementation for scraped insights.
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.domain.canonical_insight import CanonicalInsightInput
from app.repositories.canonical_insight_repository import CanonicalInsightRepository
from app.scraping.storage.base import InsightStorage


class SQLAlchemyInsightStorage(InsightStorage):
    """
    Persist canonical insights through the repository and DB session.
    """

    def __init__(self, *, session: Session, batch_size: int = 1000) -> None:
        self._session = session
        self._batch_size = max(1, batch_size)

    def store(self, rows: Sequence[CanonicalInsightInput]) -> int:
        if not rows:
            return 0

        repository = CanonicalInsightRepository(self._session)
        inserted = 0
        try:
            for start in range(0, len(rows), self._batch_size):
                chunk = rows[start : start + self._batch_size]
                inserted += repository.bulk_insert(chunk)
            self._session.commit()
            return inserted
        except SQLAlchemyError:
            self._session.rollback()
            raise
