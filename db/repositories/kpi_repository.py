"""
db/repositories/kpi_repository.py

Persistence layer for ComputedKPI records.

All methods are transaction-safe. The caller controls commit/rollback;
this repository never commits on its own.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from db.models.computed_kpi import ComputedKPI

_UPSERT_CONSTRAINT = "uq_computed_kpis_entity_period"
_DEFAULT_BATCH_SIZE = 500


class KPIRepository:
    """
    Repository for writing and querying ComputedKPI rows.

    Upsert semantics: inserting a record whose ``(entity_name, period_start,
    period_end)`` already exists replaces ``computed_kpis`` and refreshes
    ``created_at`` on conflict, rather than raising a duplicate-key error.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_kpi(
        self,
        *,
        entity_name: str,
        period_start: datetime,
        period_end: datetime,
        computed_kpis: dict[str, Any],
    ) -> ComputedKPI:
        """
        Upsert a single KPI result row.

        If a row with the same ``(entity_name, period_start, period_end)``
        already exists, ``computed_kpis`` is overwritten in place.

        Parameters
        ----------
        entity_name:
            Entity whose KPIs are stored.
        period_start:
            Inclusive start of the measurement period (timezone-aware UTC).
        period_end:
            Inclusive end of the measurement period (timezone-aware UTC).
        computed_kpis:
            Structured KPI payload, e.g.
            ``{"mrr": {"value": 1200.0, "unit": "currency"}, ...}``.

        Returns
        -------
        ComputedKPI
            The persisted ORM instance (not yet committed).
        """
        stmt = (
            insert(ComputedKPI)
            .values(
                id=uuid.uuid4(),
                entity_name=entity_name,
                period_start=period_start,
                period_end=period_end,
                computed_kpis=computed_kpis,
            )
            .on_conflict_do_update(
                constraint=_UPSERT_CONSTRAINT,
                set_={
                    "computed_kpis": computed_kpis,
                    "created_at": _now_utc(),
                },
            )
            .returning(ComputedKPI)
        )
        row: ComputedKPI = self._session.scalars(stmt).one()
        return row

    def bulk_save_kpis(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> int:
        """
        Upsert multiple KPI result rows in batches.

        Each element of ``rows`` must contain the keys:
        ``entity_name``, ``period_start``, ``period_end``, ``computed_kpis``.

        Rows with duplicate ``(entity_name, period_start, period_end)`` within
        the same call are deduplicated in Python before hitting the database;
        the last occurrence wins.

        Parameters
        ----------
        rows:
            Sequence of payload dicts.
        batch_size:
            Maximum rows per INSERT statement.

        Returns
        -------
        int
            Total number of rows written (inserted + updated).
        """
        if not rows:
            return 0

        deduped = _deduplicate(rows)
        size = max(1, batch_size)
        written = 0

        for start in range(0, len(deduped), size):
            chunk = deduped[start : start + size]
            payloads = [
                {
                    "id": uuid.uuid4(),
                    "entity_name": r["entity_name"],
                    "period_start": r["period_start"],
                    "period_end": r["period_end"],
                    "computed_kpis": r["computed_kpis"],
                }
                for r in chunk
            ]
            stmt = (
                insert(ComputedKPI)
                .values(payloads)
                .on_conflict_do_update(
                    constraint=_UPSERT_CONSTRAINT,
                    set_={
                        "computed_kpis": insert(ComputedKPI).excluded.computed_kpis,
                        "created_at": _now_utc(),
                    },
                )
                .returning(ComputedKPI.id)
            )
            written += len(self._session.scalars(stmt).all())

        return written

    def bulk_save_kpis_atomic(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> int:
        """
        Transaction-safe wrapper around :meth:`bulk_save_kpis`.

        Wraps in a savepoint when already inside a transaction so the outer
        transaction is never implicitly committed.
        """
        if not rows:
            return 0

        with self._transaction_context():
            return self.bulk_save_kpis(rows, batch_size=batch_size)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_kpis_by_period(
        self,
        *,
        period_start: datetime,
        period_end: datetime,
        entity_name: str | None = None,
    ) -> list[ComputedKPI]:
        """
        Return all ComputedKPI rows whose period window overlaps the given range.

        A row is included when its ``period_start >= period_start`` and its
        ``period_end <= period_end`` (exact-window match).  Pass
        ``entity_name`` to scope results to a single entity.

        Parameters
        ----------
        period_start:
            Lower bound of the query window (inclusive).
        period_end:
            Upper bound of the query window (inclusive).
        entity_name:
            Optional filter; when ``None``, all entities are returned.

        Returns
        -------
        list[ComputedKPI]
            Ordered by ``entity_name``, then ``period_start`` ascending.
        """
        stmt = (
            select(ComputedKPI)
            .where(
                ComputedKPI.period_start >= period_start,
                ComputedKPI.period_end <= period_end,
            )
            .order_by(ComputedKPI.entity_name, ComputedKPI.period_start)
        )

        if entity_name is not None:
            stmt = stmt.where(ComputedKPI.entity_name == entity_name)

        return list(self._session.scalars(stmt).all())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transaction_context(self) -> Any:
        if self._session.in_transaction():
            return self._session.begin_nested()
        return self._session.begin()


# ---------------------------------------------------------------------------
# Module-level helpers (no business logic)
# ---------------------------------------------------------------------------


def _deduplicate(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Last-write-wins deduplication keyed on (entity_name, period_start, period_end)."""
    seen: dict[tuple[str, datetime, datetime], dict[str, Any]] = {}
    for row in rows:
        key = (row["entity_name"], row["period_start"], row["period_end"])
        seen[key] = row
    return list(seen.values())


def _now_utc() -> datetime:
    from datetime import timezone
    return datetime.now(tz=timezone.utc)
