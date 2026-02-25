"""
app/domain/macro_ingestion.py

Domain contracts for macro data ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True)
class MacroSeriesRequest:
    """
    Request for one canonical macro series ingestion.
    """

    metric_name: str
    provider_metric: str | None = None
    period_start: str | None = None
    period_end: str | None = None
    limit: int | None = None
    unit: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MacroIngestionSummary:
    """
    Summary of one macro ingestion run.
    """

    source_key: str
    country_code: str
    run_id: UUID
    run_version: int
    fetched_records: int
    valid_records: int
    upserted_records: int
    skipped_records: int
    validation_errors: tuple[str, ...]
    ingested_at: datetime

