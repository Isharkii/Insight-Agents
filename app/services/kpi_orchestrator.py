"""
app/services/kpi_orchestrator.py

KPI pipeline orchestrator.

Wires AggregationService → KPIService → KPIRepository into a single
transactional run.  No business logic lives here; every layer retains
its own responsibility:

    AggregationService  – SQL queries, numerical extraction
    KPIService          – deterministic formula calculation
    KPIRepository       – upsert into computed_kpis

Failure contract
----------------
- Aggregation failure  → raises KPIAggregationError  (rollback implied)
- Calculation errors   → stored in the JSONB payload as ``"error"`` fields;
                         the run completes and the partial result is persisted
- Persistence failure  → raises KPIPersistenceError after rollback

The previous period window is derived automatically from the requested
period duration:

    previous_period_start = period_start - (period_end - period_start)
    previous_period_end   = period_start
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.services.aggregation_service import AggregationService
from app.services.kpi_service import (
    ChurnInput,
    GrowthRateInput,
    KPIResult,
    KPIService,
    LTVInput,
    MRRInput,
)
from db.models.computed_kpi import ComputedKPI
from db.repositories.kpi_repository import KPIRepository

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class KPIAggregationError(RuntimeError):
    """
    Raised when the aggregation layer cannot fetch inputs from the database.

    The session is left in a clean state (no partial writes occurred).
    """


class KPIPersistenceError(RuntimeError):
    """
    Raised when the computed KPI row cannot be written to ``computed_kpis``.

    The session has been rolled back before this exception is raised.
    """


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KPIRunResult:
    """
    Structured output of a single orchestrator run.

    Attributes
    ----------
    record_id:
        Primary key of the upserted ``ComputedKPI`` row.
    entity_name:
        Entity the KPIs were computed for.
    period_start:
        Inclusive start of the measurement period.
    period_end:
        Inclusive end of the measurement period.
    metrics:
        Full JSONB payload that was persisted, keyed by metric name.
        Each value is a dict with ``"value"`` (float | None),
        ``"unit"`` (str), and ``"error"`` (str | None).
    computed_at:
        UTC timestamp when the run completed.
    has_errors:
        ``True`` when at least one metric could not be computed
        (e.g. division by zero in churn or growth rate).
    """

    record_id: uuid.UUID
    entity_name: str
    period_start: datetime
    period_end: datetime
    metrics: dict[str, Any]
    computed_at: datetime
    has_errors: bool


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class KPIOrchestrator:
    """
    Coordinates the full KPI computation pipeline for a single entity and period.

    The orchestrator is **stateless** with respect to business data.
    Inject a pre-built ``KPIService`` to share the instance across runs;
    ``AggregationService`` and ``KPIRepository`` are instantiated per-call
    because they are bound to a request-scoped database session.

    Parameters
    ----------
    kpi_service:
        Shared, stateless calculation engine.  A new instance is created
        when not provided.
    """

    def __init__(self, *, kpi_service: KPIService | None = None) -> None:
        self._kpi_service = kpi_service or KPIService()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        entity_name: str,
        period_start: datetime,
        period_end: datetime,
        db: Session,
    ) -> KPIRunResult:
        """
        Execute the full KPI pipeline for *entity_name* over the given period.

        Steps
        -----
        1. Derive the previous period window from the requested duration.
        2. Fetch all aggregated inputs via ``AggregationService``.
        3. Compute MRR, Churn, LTV, and Growth Rate via ``KPIService``.
        4. Serialise results into a structured JSONB payload.
        5. Upsert the payload into ``computed_kpis`` via ``KPIRepository``.
        6. Commit the transaction.
        7. Return a :class:`KPIRunResult`.

        Parameters
        ----------
        entity_name:
            Entity whose KPIs are to be computed.
        period_start:
            Inclusive start of the measurement period (timezone-aware UTC).
        period_end:
            Inclusive end of the measurement period (timezone-aware UTC).
        db:
            Active SQLAlchemy session.  The orchestrator commits on success
            and rolls back on persistence failure.

        Raises
        ------
        KPIAggregationError
            If the aggregation layer raises a database error.
        KPIPersistenceError
            If the upsert into ``computed_kpis`` fails.
        ValueError
            If ``period_start >= period_end``.
        """
        if period_start >= period_end:
            raise ValueError(
                f"period_start must be before period_end; "
                f"got {period_start.isoformat()} >= {period_end.isoformat()}"
            )

        run_start = time.monotonic()
        logger.info(
            "KPIOrchestrator.run started entity=%r period=[%s, %s]",
            entity_name,
            period_start.isoformat(),
            period_end.isoformat(),
        )

        # Step 1 – derive previous period
        prev_start, prev_end = _previous_period(period_start, period_end)
        logger.debug(
            "Previous period derived: [%s, %s]",
            prev_start.isoformat(),
            prev_end.isoformat(),
        )

        # Step 2 – aggregate inputs
        inputs = self._aggregate(
            entity_name=entity_name,
            period_start=period_start,
            period_end=period_end,
            prev_period_start=prev_start,
            prev_period_end=prev_end,
            db=db,
        )

        # Step 3 – compute KPIs
        results = self._compute(inputs)

        # Step 4 – serialise
        payload = _build_payload(results)
        has_errors = any(v.get("error") is not None for v in payload.values())
        if has_errors:
            logger.warning(
                "KPIOrchestrator.run entity=%r: one or more metrics could not be computed; "
                "partial result will be persisted (see 'error' fields in payload)",
                entity_name,
            )

        # Step 5 & 6 – persist and commit
        record = self._persist(
            entity_name=entity_name,
            period_start=period_start,
            period_end=period_end,
            payload=payload,
            db=db,
        )

        elapsed = time.monotonic() - run_start
        computed_at = datetime.now(tz=timezone.utc)
        logger.info(
            "KPIOrchestrator.run completed entity=%r period=[%s, %s] "
            "record_id=%s has_errors=%s elapsed=%.3fs",
            entity_name,
            period_start.isoformat(),
            period_end.isoformat(),
            record.id,
            has_errors,
            elapsed,
        )

        return KPIRunResult(
            record_id=record.id,
            entity_name=entity_name,
            period_start=period_start,
            period_end=period_end,
            metrics=payload,
            computed_at=computed_at,
            has_errors=has_errors,
        )

    # ------------------------------------------------------------------
    # Internal: aggregation
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        *,
        entity_name: str,
        period_start: datetime,
        period_end: datetime,
        prev_period_start: datetime,
        prev_period_end: datetime,
        db: Session,
    ) -> _AggregatedInputs:
        """
        Fetch all numerical inputs required by KPIService in one coordinated pass.

        Raises
        ------
        KPIAggregationError
            Wraps any ``SQLAlchemyError`` raised by the aggregation layer.
        """
        agg = AggregationService(db)

        try:
            t0 = time.monotonic()

            subscription_revenues = agg.get_subscription_revenues(
                entity_name, period_start, period_end
            )
            logger.debug(
                "_aggregate subscription_revenues count=%d",
                len(subscription_revenues),
            )

            active_customers = agg.get_active_customers(entity_name, period_start)
            logger.debug("_aggregate active_customers=%d", active_customers)

            lost_customers = agg.get_lost_customers(
                entity_name, period_start, period_end
            )
            logger.debug("_aggregate lost_customers=%d", lost_customers)

            arpu = agg.get_average_revenue_per_user(
                entity_name, period_start, period_end
            )
            logger.debug("_aggregate arpu=%.4f", arpu)

            current_revenue = agg.get_period_revenue(
                entity_name, period_start, period_end
            )
            logger.debug("_aggregate current_revenue=%.4f", current_revenue)

            previous_revenue = agg.get_period_revenue(
                entity_name, prev_period_start, prev_period_end
            )
            logger.debug("_aggregate previous_revenue=%.4f", previous_revenue)

            logger.debug(
                "_aggregate complete entity=%r elapsed=%.3fs",
                entity_name,
                time.monotonic() - t0,
            )

            return _AggregatedInputs(
                subscription_revenues=subscription_revenues,
                active_customers=active_customers,
                lost_customers=lost_customers,
                arpu=arpu,
                current_revenue=current_revenue,
                previous_revenue=previous_revenue,
            )

        except SQLAlchemyError as exc:
            logger.error(
                "_aggregate failed entity=%r: %s",
                entity_name,
                exc,
                exc_info=True,
            )
            raise KPIAggregationError(
                f"Failed to aggregate KPI inputs for entity={entity_name!r}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Internal: calculation
    # ------------------------------------------------------------------

    def _compute(self, inputs: _AggregatedInputs) -> _KPIResults:
        """
        Run all four KPI calculations against the pre-fetched inputs.

        Each metric is computed independently.  A failure in one (e.g. churn
        returning None due to zero customers) does not block the others.
        LTV uses the churn result — when churn is undefinable, LTV receives
        ``churn_rate=0.0`` which triggers its own division-by-zero guard.
        """
        mrr = self._kpi_service.calculate_mrr(
            MRRInput(active_subscription_revenues=inputs.subscription_revenues)
        )
        logger.debug("_compute mrr value=%s error=%s", mrr.value, mrr.error)

        churn = self._kpi_service.calculate_churn(
            ChurnInput(
                customers_at_start=inputs.active_customers,
                customers_lost=inputs.lost_customers,
            )
        )
        logger.debug("_compute churn value=%s error=%s", churn.value, churn.error)

        # Feed churn_rate to LTV; fall back to 0.0 so KPIService handles the
        # undefined case rather than raising here.
        churn_rate_for_ltv = churn.value if churn.value is not None else 0.0
        ltv = self._kpi_service.calculate_ltv(
            LTVInput(
                average_revenue_per_user=inputs.arpu,
                churn_rate=churn_rate_for_ltv,
            )
        )
        logger.debug("_compute ltv value=%s error=%s", ltv.value, ltv.error)

        growth = self._kpi_service.calculate_growth_rate(
            GrowthRateInput(
                current_period_revenue=inputs.current_revenue,
                previous_period_revenue=inputs.previous_revenue,
            )
        )
        logger.debug("_compute growth_rate value=%s error=%s", growth.value, growth.error)

        return _KPIResults(mrr=mrr, churn=churn, ltv=ltv, growth=growth)

    # ------------------------------------------------------------------
    # Internal: persistence
    # ------------------------------------------------------------------

    def _persist(
        self,
        *,
        entity_name: str,
        period_start: datetime,
        period_end: datetime,
        payload: dict[str, Any],
        db: Session,
    ) -> ComputedKPI:
        """
        Upsert the payload and commit the session.

        Raises
        ------
        KPIPersistenceError
            If the upsert or commit fails.  The session is rolled back before
            the exception propagates.
        """
        repository = KPIRepository(db)
        try:
            record = repository.save_kpi(
                entity_name=entity_name,
                period_start=period_start,
                period_end=period_end,
                computed_kpis=payload,
            )
            db.commit()
            logger.debug(
                "_persist upserted entity=%r record_id=%s",
                entity_name,
                record.id,
            )
            return record

        except SQLAlchemyError as exc:
            db.rollback()
            logger.error(
                "_persist failed entity=%r: %s",
                entity_name,
                exc,
                exc_info=True,
            )
            raise KPIPersistenceError(
                f"Failed to persist KPI results for entity={entity_name!r}: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Internal data containers (not part of public API)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _AggregatedInputs:
    """Raw numerical values fetched from the aggregation layer."""

    subscription_revenues: list[float]
    active_customers: int
    lost_customers: int
    arpu: float
    current_revenue: float
    previous_revenue: float


@dataclass(frozen=True)
class _KPIResults:
    """KPIResult objects for all four metrics."""

    mrr: KPIResult
    churn: KPIResult
    ltv: KPIResult
    growth: KPIResult


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _previous_period(
    period_start: datetime,
    period_end: datetime,
) -> tuple[datetime, datetime]:
    """
    Derive the immediately preceding period of equal duration.

    The previous period ends exactly at ``period_start`` and has the same
    length as ``[period_start, period_end]``::

        prev_end   = period_start
        prev_start = period_start - (period_end - period_start)
    """
    duration = period_end - period_start
    return period_start - duration, period_start


def _build_payload(results: _KPIResults) -> dict[str, Any]:
    """
    Serialise ``_KPIResults`` into the JSONB structure stored in ``computed_kpis``.

    Shape::

        {
            "mrr":         {"value": 12000.0, "unit": "currency", "error": null},
            "churn_rate":  {"value": 0.05,    "unit": "rate",     "error": null},
            "ltv":         {"value": 2400.0,  "unit": "currency", "error": null},
            "growth_rate": {"value": 0.12,    "unit": "rate",     "error": null}
        }
    """
    return {
        result.metric: {
            "value": result.value,
            "unit": result.unit,
            "error": result.error,
        }
        for result in (results.mrr, results.churn, results.ltv, results.growth)
    }
