"""
app/domain/canonical_insight.py

Domain models used by CSV ingestion flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class CanonicalInsightInput:
    """
    Typed canonical record prepared for persistence.
    """

    source_type: str
    entity_name: str
    category: str
    metric_name: str
    metric_value: Any
    timestamp: datetime
    region: str | None
    metadata_json: dict[str, Any] | None


@dataclass(frozen=True)
class RowValidationError:
    """
    One CSV row validation error detail.
    """

    row_number: int
    message: str
    column: str | None = None
    value: str | None = None


@dataclass(frozen=True)
class IngestionSummary:
    """
    End-of-run ingestion summary.
    """

    rows_processed: int
    rows_failed: int
    validation_errors: list[RowValidationError] = field(default_factory=list)
