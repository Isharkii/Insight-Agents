"""
app/services/dataset_hash.py

Deterministic SHA-256 hash of canonical insight records for a given
entity and period.  Used by the analytics guard to detect whether the
underlying dataset has changed since the last KPI computation.
"""

from __future__ import annotations

import hashlib
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models.canonical_insight_record import CanonicalInsightRecord


def compute_dataset_hash(
    db: Session,
    entity_name: str,
    category_aliases: tuple[str, ...],
    period_start: datetime,
    period_end: datetime,
) -> str:
    """
    Compute a SHA-256 hex digest over all canonical rows for the entity/period.

    The query is ordered deterministically by (metric_name, timestamp) so
    the same dataset always produces the same hash regardless of insertion
    order.

    Returns
    -------
    str
        64-character hex digest.  If no rows match, returns the SHA-256
        of the empty string (deterministic sentinel for "no data").
    """
    stmt = (
        select(
            CanonicalInsightRecord.metric_name,
            CanonicalInsightRecord.metric_value,
            CanonicalInsightRecord.timestamp,
        )
        .where(
            CanonicalInsightRecord.entity_name == entity_name,
            CanonicalInsightRecord.category.in_(category_aliases),
            CanonicalInsightRecord.timestamp >= period_start,
            CanonicalInsightRecord.timestamp <= period_end,
        )
        .order_by(
            CanonicalInsightRecord.metric_name,
            CanonicalInsightRecord.timestamp,
        )
    )

    hasher = hashlib.sha256()
    for metric_name, metric_value, ts in db.execute(stmt):
        line = f"{metric_name}|{metric_value}|{ts.isoformat()}\n"
        hasher.update(line.encode("utf-8"))

    return hasher.hexdigest()
