"""
db/repositories/macro_metrics_repository.py

Persistence helpers for macro metric ingestion with idempotent upserts.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import date, datetime
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from db.models.macro_metrics import MacroMetric, MacroMetricRun

_RUN_UPSERT_CONSTRAINT = "uq_macro_metric_runs_source_country_version"
_METRIC_UPSERT_CONSTRAINT = "uq_macro_metrics_run_country_metric_frequency_period_end"


class MacroMetricsRepository:
    """
    Repository for versioned macro time-series ingestion writes.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def next_run_version(
        self,
        *,
        source_key: str,
        country_code: str,
    ) -> int:
        stmt = select(func.max(MacroMetricRun.run_version)).where(
            MacroMetricRun.source_key == source_key,
            MacroMetricRun.country_code == country_code,
        )
        latest = self._session.scalar(stmt)
        return int(latest or 0) + 1

    def upsert_run(
        self,
        *,
        source_key: str,
        country_code: str,
        run_version: int,
        source_release_ts: datetime | None,
        is_current: bool,
        metadata_json: dict[str, Any] | None = None,
    ) -> uuid.UUID:
        stmt = (
            insert(MacroMetricRun)
            .values(
                id=uuid.uuid4(),
                source_key=source_key,
                country_code=country_code,
                run_version=run_version,
                source_release_ts=source_release_ts,
                is_current=is_current,
                metadata_json=metadata_json,
            )
            .on_conflict_do_update(
                constraint=_RUN_UPSERT_CONSTRAINT,
                set_={
                    "source_release_ts": source_release_ts,
                    "is_current": is_current,
                    "metadata_json": metadata_json,
                    "ingested_at": func.now(),
                },
            )
            .returning(MacroMetricRun.id)
        )
        return self._session.scalars(stmt).one()

    def mark_only_current(
        self,
        *,
        source_key: str,
        country_code: str,
        current_run_id: uuid.UUID,
    ) -> int:
        stmt = (
            update(MacroMetricRun)
            .where(
                MacroMetricRun.source_key == source_key,
                MacroMetricRun.country_code == country_code,
                MacroMetricRun.id != current_run_id,
                MacroMetricRun.is_current.is_(True),
            )
            .values(is_current=False)
        )
        return int(self._session.execute(stmt).rowcount or 0)

    def bulk_upsert_metrics(
        self,
        *,
        run_id: uuid.UUID,
        rows: Sequence[dict[str, Any]],
        batch_size: int = 500,
    ) -> int:
        if not rows:
            return 0

        deduped = _deduplicate_rows(rows)
        size = max(1, batch_size)
        written = 0

        for start in range(0, len(deduped), size):
            chunk = deduped[start : start + size]
            payloads = [
                {
                    "id": uuid.uuid4(),
                    "run_id": run_id,
                    "country_code": str(row["country_code"]).upper(),
                    "metric_name": str(row["metric_name"]),
                    "frequency": str(row["frequency"]),
                    "period_start": row["period_start"],
                    "period_end": row["period_end"],
                    "value": float(row["value"]),
                    "unit": row.get("unit"),
                    "metadata_json": row.get("metadata_json"),
                }
                for row in chunk
            ]

            stmt = (
                insert(MacroMetric)
                .values(payloads)
                .on_conflict_do_update(
                    constraint=_METRIC_UPSERT_CONSTRAINT,
                    set_={
                        "period_start": insert(MacroMetric).excluded.period_start,
                        "value": insert(MacroMetric).excluded.value,
                        "unit": insert(MacroMetric).excluded.unit,
                        "metadata_json": insert(MacroMetric).excluded.metadata_json,
                    },
                )
                .returning(MacroMetric.id)
            )
            written += len(self._session.scalars(stmt).all())

        return written


def _deduplicate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Last-write-wins deduplication by macro series key.
    """

    deduped: dict[tuple[str, str, str, date], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["country_code"]).upper(),
            str(row["metric_name"]).lower(),
            str(row["frequency"]).upper(),
            row["period_end"],
        )
        deduped[key] = dict(row)
    return list(deduped.values())

