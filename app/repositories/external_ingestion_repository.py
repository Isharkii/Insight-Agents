"""
app/repositories/external_ingestion_repository.py

DB persistence for external connector records.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy.orm import Session

from app.domain.canonical_insight import CanonicalInsightInput
from db.models.canonical_insight_record import CanonicalInsightRecord


class ExternalIngestionRepository:
    """
    Repository responsible for storing normalized external records.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def bulk_insert_records(
        self,
        records: Sequence[CanonicalInsightInput],
        *,
        batch_size: int = 1000,
    ) -> int:
        """
        Insert canonical records in configurable chunks.
        """

        if not records:
            return 0

        size = max(1, batch_size)
        inserted = 0
        for start in range(0, len(records), size):
            chunk = records[start : start + size]
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
                for row in chunk
            ]
            self._session.bulk_insert_mappings(CanonicalInsightRecord, payloads)
            inserted += len(payloads)
        return inserted
