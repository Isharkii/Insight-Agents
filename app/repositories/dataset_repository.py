"""
app/repositories/dataset_repository.py

Read-only queries for inspecting ingested canonical data.

Provides convenience methods for the analyze endpoint and other callers
that need to discover entity names, categories, or date ranges from
the canonical_insight_records table without coupling to analytics engines.
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from db.models.canonical_insight_record import CanonicalInsightRecord


class DatasetRepository:
    """
    Lightweight read-only repository over canonical_insight_records.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def get_distinct_entities(self) -> list[str]:
        """Return all unique entity_name values, ordered alphabetically."""
        stmt = (
            select(CanonicalInsightRecord.entity_name)
            .distinct()
            .order_by(CanonicalInsightRecord.entity_name)
        )
        return list(self._session.scalars(stmt).all())

    def get_distinct_categories(self, *, entity_name: str | None = None) -> list[str]:
        """Return unique category values, optionally filtered by entity."""
        stmt = select(CanonicalInsightRecord.category).distinct()
        if entity_name:
            stmt = stmt.where(CanonicalInsightRecord.entity_name == entity_name)
        stmt = stmt.order_by(CanonicalInsightRecord.category)
        return list(self._session.scalars(stmt).all())

    def get_latest_entity(self) -> str | None:
        """Return the entity_name from the most recently ingested record."""
        stmt = (
            select(CanonicalInsightRecord.entity_name)
            .order_by(CanonicalInsightRecord.created_at.desc())
            .limit(1)
        )
        return self._session.scalar(stmt)

    def get_record_count(
        self,
        *,
        entity_name: str | None = None,
        category: str | None = None,
    ) -> int:
        """Count records, optionally filtered by entity and/or category."""
        stmt = select(func.count(CanonicalInsightRecord.id))
        if entity_name:
            stmt = stmt.where(CanonicalInsightRecord.entity_name == entity_name)
        if category:
            stmt = stmt.where(CanonicalInsightRecord.category == category)
        return self._session.scalar(stmt) or 0

    def get_date_range(
        self,
        *,
        entity_name: str | None = None,
    ) -> tuple[datetime | None, datetime | None]:
        """Return (earliest, latest) timestamp for the given entity or all data."""
        stmt = select(
            func.min(CanonicalInsightRecord.timestamp),
            func.max(CanonicalInsightRecord.timestamp),
        )
        if entity_name:
            stmt = stmt.where(CanonicalInsightRecord.entity_name == entity_name)
        row = self._session.execute(stmt).one_or_none()
        if row is None:
            return None, None
        return row[0], row[1]

    def get_metrics_for_entity(
        self,
        entity_name: str,
        *,
        category: str | None = None,
        limit: int = 1000,
    ) -> Sequence[CanonicalInsightRecord]:
        """Fetch metric records for an entity, ordered by timestamp desc."""
        stmt = (
            select(CanonicalInsightRecord)
            .where(CanonicalInsightRecord.entity_name == entity_name)
            .order_by(CanonicalInsightRecord.timestamp.desc())
            .limit(limit)
        )
        if category:
            stmt = stmt.where(CanonicalInsightRecord.category == category)
        return self._session.scalars(stmt).all()
