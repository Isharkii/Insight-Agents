"""
app/repositories/canonical_insight_repository.py

Persistence layer for canonical insight records.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy.orm import Session

from app.domain.canonical_insight import CanonicalInsightInput
from db.models.canonical_insight_record import CanonicalInsightRecord


class CanonicalInsightRepository:
    """
    Repository for batch persistence of canonical insight records.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def bulk_insert(self, rows: Sequence[CanonicalInsightInput]) -> int:
        """
        Insert a batch of canonical rows using ORM bulk mappings.
        """

        if not rows:
            return 0

        payloads: list[dict[str, Any]] = [
            {
                "source_type": row.source_type,
                "entity_name": row.entity_name,
                "category": row.category,
                "metric_name": row.metric_name,
                "metric_value": row.metric_value,
                "timestamp": row.timestamp,
                "region": row.region,
                "metadata_json": row.metadata_json,
            }
            for row in rows
        ]

        self._session.bulk_insert_mappings(CanonicalInsightRecord, payloads)
        return len(payloads)
