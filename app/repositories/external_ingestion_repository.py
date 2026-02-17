"""
app/repositories/external_ingestion_repository.py

DB persistence for external connector records.
"""

from __future__ import annotations

from collections.abc import Sequence
from sqlalchemy.orm import Session

from app.domain.canonical_insight import CanonicalInsightInput
from app.repositories.canonical_insight_repository import CanonicalInsightRepository


class ExternalIngestionRepository:
    """
    Repository responsible for storing normalized external records.
    """

    def __init__(self, session: Session) -> None:
        self._session = session
        self._canonical_repository = CanonicalInsightRepository(session)

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

        return self._canonical_repository.bulk_insert(records, batch_size=batch_size)
