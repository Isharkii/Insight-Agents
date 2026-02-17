"""
app/services/kpi_orchestrator.py

KPI pipeline orchestrator.

Wires AggregationService → KPI formula → KPIRepository into a single
transactional run.  No business logic lives here; every layer retains
its own responsibility:

    AggregationService  – SQL queries, numerical extraction
    KPI formula         – deterministic formula calculation (routed by business_type)
    KPIRepository       – upsert into computed_kpis

Business-type routing
---------------------
business_type="saas"       → SaaSKPIFormula
business_type="ecommerce"  → EcommerceKPIFormula
business_type="agency"     → AgencyKPIFormula

Failure contract
----------------
- Unknown business_type    → raises KPIUnknownBusinessTypeError immediately
- Aggregation failure      → raises KPIAggregationError  (rollback implied)
- Calculation errors       → stored in the JSONB payload as ``"error"`` fields;
                             the run completes and the partial result is persisted
- Persistence failure      → raises KPIPersistenceError after rollback

The previous period window is derived automatically from the requested
period duration:

    previous_period_start = period_start - (period_end - period_start)
    previous_period_end   = period_start

Extra inputs
------------
Aggregated DB inputs cover the core numerical fields common to all business
types.  Fields that cannot be derived from ``CanonicalInsightRecord``
(e.g. ``gross_margin``, ``total_visitors``, ``billable_hours``) must be
supplied via the ``extra_inputs`` parameter of :meth:`KPIOrchestrator.run`.
Missing extra inputs default to zero/empty so that the formula's own
division-by-zero guards produce ``None`` values rather than raising.
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
from db.models.computed_kpi import ComputedKPI
from db.repositories.kpi_repository import KPIRepository
from kpi.agency import AgencyKPIFormula
from kpi.base import BaseKPIFormula
from kpi.ecommerce import EcommerceKPIFormula
from kpi.saas import SaaSKPIFormula

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Formula registry
# ---------------------------------------------------------------------------

_FORMULA_REGISTRY: dict[str, BaseKPIFormula] = {
    "saas": SaaSKPIFormula(),
    "ecommerce": EcommerceKPIFormula(),
    "agency": AgencyKPIFormula(),
}

# Metrics whose natural unit is a dimensionless rate rather than currency.
_RATE_METRICS: frozenset[str] = frozenset(
    {
        "churn_rate",
        "client_churn",
        "conversion_rate",
        "utilization_rate",
        "growth_rate",
        "purchase_frequency",
    }
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class KPIUnknownBusinessTypeError(ValueError):
    """
    Raised immediately when an unrecognised ``business_type`` is passed.

    Valid values are the keys of :data:`_FORMULA_REGISTRY`.
    """


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
    business_type:
        Formula family that was used (``"saas"``, ``"ecommerce"``, ``"agency"``).
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
        (e.g. division by zero).
    """

    record_id: uuid.UUID
    entity_name: str
    business_type: str
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
    ``AggregationService`` and ``KPIRepository`` are instantiated per-call
    because they are bound to a request-scoped database session.
    """

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        entity_name: str,
        business_type: str,
        period_start: datetime,
        period_end: datetime,
        db: Session,
        extra_inputs: dict[str, Any] | None = None,
    ) -> KPIRunResult:
        """
        Execute the full KPI pipeline for *entity_name* over the given period.

        Steps
        -----
        1. Validate ``business_type`` against the formula registry.
        2. Validate that ``period_start < period_end``.
        3. Derive the previous period window from the requested duration.
        4. Fetch all aggregated inputs via ``AggregationService``.
        5. Merge aggregated inputs with ``extra_inputs`` and call the formula.
        6. Serialise results into a structured JSONB payload.
        7. Upsert the payload into ``computed_kpis`` via ``KPIRepository``.
        8. Commit the transaction.
        9. Return a :class:`KPIRunResult`.

        Parameters
        ----------
        entity_name:
            Entity whose KPIs are to be computed.
        business_type:
            One of ``"saas"``, ``"ecommerce"``, or ``"agency"``.
        period_start:
            Inclusive start of the measurement period (timezone-aware UTC).
        period_end:
            Inclusive end of the measurement period (timezone-aware UTC).
        db:
            Active SQLAlchemy session.  The orchestrator commits on success
            and rolls back on persistence failure.
        extra_inputs:
            Optional dict of business-type-specific values that cannot be
            derived from the database (e.g. ``gross_margin``,
            ``total_visitors``, ``billable_hours``).  Missing keys default
            to zero so the formula's division-by-zero guards apply.

        Raises
        ------
        KPIUnknownBusinessTypeError
            If ``business_type`` is not registered.
        KPIAggregationError
            If the aggregation layer raises a database error.
        KPIPersistenceError
            If the upsert into ``computed_kpis`` fails.
        ValueError
            If ``period_start >= period_end``.
        """
        # Step 1 – validate business type early (cheap, no DB required)
        if business_type not in _FORMULA_REGISTRY:
            raise KPIUnknownBusinessTypeError(
                f"Unknown business_type {business_type!r}. "
                f"Valid types: {sorted(_FORMULA_REGISTRY)}"
            )

        # Step 2 – validate period
        if period_start >= period_end:
            raise ValueError(
                f"period_start must be before period_end; "
                f"got {period_start.isoformat()} >= {period_end.isoformat()}"
            )

        run_start = time.monotonic()
        logger.info(
            "KPIOrchestrator.run started entity=%r business_type=%r period=[%s, %s]",
            entity_name,
            business_type,
            period_start.isoformat(),
            period_end.isoformat(),
        )

        # Step 3 – derive previous period
        prev_start, prev_end = _previous_period(period_start, period_end)
        logger.debug(
            "Previous period derived: [%s, %s]",
            prev_start.isoformat(),
            prev_end.isoformat(),
        )

        # Step 4 – aggregate DB inputs
        agg_inputs = self._aggregate(
            entity_name=entity_name,
            period_start=period_start,
            period_end=period_end,
            prev_period_start=prev_start,
            prev_period_end=prev_end,
            db=db,
        )

        # Step 5 – compute KPIs via the selected formula
        metrics = self._compute(
            business_type=business_type,
            agg_inputs=agg_inputs,
            extra_inputs=extra_inputs or {},
        )

        # Step 6 – serialise
        payload = _build_payload(metrics)
        has_errors = any(v.get("error") is not None for v in payload.values())
        if has_errors:
            logger.warning(
                "KPIOrchestrator.run entity=%r business_type=%r: "
                "one or more metrics could not be computed; "
                "partial result will be persisted (see 'error' fields in payload)",
                entity_name,
                business_type,
            )

        # Step 7 & 8 – persist and commit
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
            "KPIOrchestrator.run completed entity=%r business_type=%r "
            "period=[%s, %s] record_id=%s has_errors=%s elapsed=%.3fs",
            entity_name,
            business_type,
            period_start.isoformat(),
            period_end.isoformat(),
            record.id,
            has_errors,
            elapsed,
        )

        return KPIRunResult(
            record_id=record.id,
            entity_name=entity_name,
            business_type=business_type,
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
        Fetch all numerical inputs available from the database in one pass.

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

    def _compute(
        self,
        *,
        business_type: str,
        agg_inputs: _AggregatedInputs,
        extra_inputs: dict[str, Any],
    ) -> dict[str, float | None]:
        """
        Build the formula input dict and invoke the registered formula.

        Aggregated DB values are mapped to each formula's expected keys.
        Fields that cannot be derived from the database are taken from
        ``extra_inputs``; absent keys default to zero so the formula's own
        division-by-zero guards emit ``None`` rather than raising.

        Parameters
        ----------
        business_type:
            Key into :data:`_FORMULA_REGISTRY`.
        agg_inputs:
            Numerical values fetched from ``CanonicalInsightRecord``.
        extra_inputs:
            Caller-supplied values for fields not available in the DB.

        Returns
        -------
        dict[str, float | None]
            Raw metric values keyed by metric name.
        """
        formula_inputs = _build_formula_inputs(business_type, agg_inputs, extra_inputs)
        formula = _FORMULA_REGISTRY[business_type]
        logger.debug("_compute invoking formula for business_type=%r", business_type)
        metrics = formula.calculate(formula_inputs)
        logger.debug("_compute metrics=%s", metrics)
        return metrics

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


def _build_formula_inputs(
    business_type: str,
    inputs: _AggregatedInputs,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """
    Map aggregated DB values and caller-supplied extras to the dict shape
    expected by each formula's ``calculate()`` method.

    Any extra key absent from *extra* defaults to ``0`` / ``0.0`` / ``[]``
    so the formula's division-by-zero guards apply cleanly.
    """
    if business_type == "saas":
        return {
            "active_subscriptions": inputs.subscription_revenues,
            "starting_customers": inputs.active_customers,
            "lost_customers": inputs.lost_customers,
            "gross_margin": extra.get("gross_margin", 0.0),
            "previous_mrr": inputs.previous_revenue,
        }

    if business_type == "ecommerce":
        return {
            "orders": inputs.subscription_revenues,
            "total_visitors": extra.get("total_visitors", 0),
            "marketing_spend": extra.get("marketing_spend", 0.0),
            "new_customers": extra.get("new_customers", 0),
            "unique_customers": inputs.active_customers,
            "previous_revenue": inputs.previous_revenue,
        }

    if business_type == "agency":
        return {
            "retainer_fees": inputs.subscription_revenues,
            "project_values": extra.get("project_values", []),
            "starting_clients": inputs.active_customers,
            "lost_clients": inputs.lost_customers,
            "billable_hours": extra.get("billable_hours", 0.0),
            "available_hours": extra.get("available_hours", 0.0),
            "total_employees": extra.get("total_employees", 0),
            "average_client_lifespan_months": extra.get(
                "average_client_lifespan_months", 0
            ),
        }

    # Registry guard: this branch is unreachable in normal operation because
    # business_type is validated against _FORMULA_REGISTRY before _compute()
    # is called.  Included defensively.
    raise KPIUnknownBusinessTypeError(  # pragma: no cover
        f"No input mapping defined for business_type={business_type!r}"
    )


def _build_payload(metrics: dict[str, float | None]) -> dict[str, Any]:
    """
    Serialise formula output into the JSONB structure stored in ``computed_kpis``.

    Shape::

        {
            "mrr":         {"value": 12000.0, "unit": "currency", "error": null},
            "churn_rate":  {"value": 0.05,    "unit": "rate",     "error": null},
            "ltv":         {"value": None,    "unit": "currency", "error": "division_by_zero"},
        }

    The ``"unit"`` is ``"rate"`` for metrics listed in :data:`_RATE_METRICS`
    and ``"currency"`` for all others.
    """
    return {
        name: {
            "value": value,
            "unit": "rate" if name in _RATE_METRICS else "currency",
            "error": "division_by_zero" if value is None else None,
        }
        for name, value in metrics.items()
    }
