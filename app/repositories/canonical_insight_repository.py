"""
app/repositories/canonical_insight_repository.py

Persistence layer for canonical insight records.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.domain.canonical_insight import CanonicalInsightInput
from db.models.canonical_insight_record import CanonicalInsightRecord

_DEFAULT_BATCH_SIZE = 1000
_DEDUPE_CONSTRAINT = "uq_canonical_insight_records_dedupe"


class CanonicalInsightRepository:
    """
    Repository for batch persistence of canonical insight records.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def bulk_insert(
        self,
        rows: Sequence[CanonicalInsightInput],
        *,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> int:
        """
        Insert canonical rows with PostgreSQL bulk INSERT + deduplication.
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
        return self._bulk_insert_payloads(payloads, batch_size=batch_size)

    def bulk_insert_atomic(
        self,
        rows: Sequence[CanonicalInsightInput],
        *,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> int:
        """
        Transaction-safe wrapper for bulk_insert.
        """

        if not rows:
            return 0

        with self._transaction_context():
            return self.bulk_insert(rows, batch_size=batch_size)

    def bulk_insert_models(
        self,
        rows: Sequence[CanonicalInsightRecord],
        *,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> int:
        """
        Insert materialized model rows with PostgreSQL bulk INSERT + deduplication.
        """

        if not rows:
            return 0

        payloads: list[dict[str, Any]] = []
        for row in rows:
            payload: dict[str, Any] = {
                "source_type": row.source_type,
                "entity_name": row.entity_name,
                "category": row.category,
                "metric_name": row.metric_name,
                "metric_value": row.metric_value,
                "timestamp": row.timestamp,
                "region": row.region,
                "metadata_json": row.metadata_json,
            }
            if row.id is not None:
                payload["id"] = row.id
            payloads.append(payload)

        return self._bulk_insert_payloads(payloads, batch_size=batch_size)

    def bulk_insert_models_atomic(
        self,
        rows: Sequence[CanonicalInsightRecord],
        *,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> int:
        """
        Transaction-safe wrapper for bulk_insert_models.
        """

        if not rows:
            return 0

        with self._transaction_context():
            return self.bulk_insert_models(rows, batch_size=batch_size)

    def _bulk_insert_payloads(
        self,
        payloads: Sequence[dict[str, Any]],
        *,
        batch_size: int,
    ) -> int:
        size = max(1, batch_size)
        deduped_payloads = self._deduplicate_payloads(payloads)
        inserted = 0

        for start in range(0, len(deduped_payloads), size):
            chunk = deduped_payloads[start : start + size]
            stmt = (
                insert(CanonicalInsightRecord)
                .values(chunk)
                .on_conflict_do_nothing(constraint=_DEDUPE_CONSTRAINT)
                .returning(CanonicalInsightRecord.id)
            )
            inserted += len(self._session.scalars(stmt).all())

        return inserted

    def _deduplicate_payloads(
        self,
        payloads: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        seen: set[tuple[str, str, str, str, Any]] = set()
        deduped_payloads: list[dict[str, Any]] = []

        for payload in payloads:
            key = (
                payload["source_type"],
                payload["entity_name"],
                payload["category"],
                payload["metric_name"],
                payload["timestamp"],
            )
            if key in seen:
                continue
            seen.add(key)
            deduped_payloads.append(payload)

        return deduped_payloads

    def _transaction_context(self) -> Any:
        if self._session.in_transaction():
            return self._session.begin_nested()
        return self._session.begin()
