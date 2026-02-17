"""
app/domain/external_ingestion.py

Domain models for external ingestion orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceIngestionSummary:
    """
    Summary for one external source ingestion run.
    """

    source: str
    records_inserted: int
    failed_records: int
