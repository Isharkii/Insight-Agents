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

import copy
import json
import logging
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.services.aggregation_service import AggregationService
from app.services.canonical_validation import validate_canonical_inputs_for_kpi
from app.services.category_registry import (
    CategoryRegistryError,
    CategoryPack,
    DependencyRule,
    require_category_pack,
    supported_categories,
)
from app.services.kpi_canonical_schema import (
    category_aliases_for_business_type,
    metric_aliases_for_business_type,
)
from db.models.computed_kpi import ComputedKPI
from db.repositories.kpi_repository import KPIRepository

logger = logging.getLogger(__name__)

# Bump this constant whenever formula logic, aggregation queries, or the
# pipeline structure change.  Existing rows with a different (or NULL)
# version will be recomputed on the next analysis run.
ANALYTICS_VERSION: int = 2

_DEFAULT_RATE_METRICS: frozenset[str] = frozenset(
    {"churn_rate", "client_churn", "conversion_rate", "utilization_rate", "growth_rate"}
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class KPIUnknownBusinessTypeError(ValueError):
    """
    Raised immediately when an unrecognised ``business_type`` is passed.

    Valid values are loaded from the category registry packs.
    """


class KPIAggregationError(RuntimeError):
    """
    Raised when the aggregation layer cannot fetch inputs from the database.

    The session is left in a clean state (no partial writes occurred).
    """


class KPICanonicalValidationError(KPIAggregationError):
    """
    Raised when canonical records fail pre-aggregation integrity validation.
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        super().__init__(json.dumps(payload))


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
        dataset_hash: str | None = None,
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
        try:
            pack = require_category_pack(business_type)
        except CategoryRegistryError as exc:
            raise KPIUnknownBusinessTypeError(
                f"Unknown business_type {business_type!r}. "
                f"Valid types: {list(supported_categories())}"
            ) from exc
        business_type = pack.name

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

        # Step 3 – validate canonical integrity before aggregation
        canonical_validation = validate_canonical_inputs_for_kpi(
            db=db,
            entity_name=entity_name,
            business_type=business_type,
            period_start=period_start,
            period_end=period_end,
        )
        if not canonical_validation.is_valid:
            payload = canonical_validation.error_payload or {
                "error_type": "canonical_validation_failed",
                "missing_metrics": canonical_validation.missing_metrics,
            }
            if canonical_validation.diagnostics:
                payload["diagnostics"] = canonical_validation.diagnostics
            logger.warning(
                "KPI canonical validation failed entity=%r business_type=%r "
                "missing_metrics=%s diagnostics=%s",
                entity_name,
                business_type,
                payload.get("missing_metrics"),
                canonical_validation.diagnostics,
            )
            raise KPICanonicalValidationError(payload)

        # Step 4 – derive previous period
        prev_start, prev_end = _previous_period(period_start, period_end)
        logger.debug(
            "Previous period derived: [%s, %s]",
            prev_start.isoformat(),
            prev_end.isoformat(),
        )

        # Step 5 – aggregate DB inputs
        agg_inputs = self._aggregate(
            entity_name=entity_name,
            business_type=business_type,
            period_start=period_start,
            period_end=period_end,
            prev_period_start=prev_start,
            prev_period_end=prev_end,
            db=db,
        )

        # Step 6 – compute KPIs via the selected formula
        metrics, validity = self._compute(
            business_type=business_type,
            agg_inputs=agg_inputs,
            extra_inputs=extra_inputs or {},
        )

        # Step 6b – backfill None metrics from precomputed canonical values
        metrics, validity = self._backfill_from_precomputed(
            metrics=metrics,
            validity=validity,
            entity_name=entity_name,
            period_start=period_start,
            period_end=period_end,
            pack=pack,
            db=db,
        )

        # Step 7 – serialise (includes validity flags for LLM consumption)
        payload = _build_payload(
            metrics,
            validity,
            rate_metrics=pack.rate_metrics or _DEFAULT_RATE_METRICS,
        )
        has_errors = any(v.get("error") is not None for v in payload.values())
        if has_errors:
            logger.warning(
                "KPIOrchestrator.run entity=%r business_type=%r: "
                "one or more metrics could not be computed; "
                "partial result will be persisted (see 'error' fields in payload)",
                entity_name,
                business_type,
            )

        # Step 8 & 9 – persist and commit
        record = self._persist(
            entity_name=entity_name,
            period_start=period_start,
            period_end=period_end,
            payload=payload,
            db=db,
            analytics_version=ANALYTICS_VERSION,
            dataset_hash=dataset_hash,
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
        business_type: str,
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
        metric_aliases = metric_aliases_for_business_type(business_type)
        agg = AggregationService(
            db,
            metric_recurring_revenue=metric_aliases["recurring_revenue"],
            metric_active_customer_count=metric_aliases["active_customer_count"],
            metric_churned_customer_count=metric_aliases["churned_customer_count"],
            categories=category_aliases_for_business_type(business_type),
        )

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
    ) -> tuple[dict[str, float | None], dict[str, dict[str, Any]]]:
        """
        Build the formula input dict, invoke the registered formula, and
        collect per-metric validity information.

        Aggregated DB values are mapped to each formula's expected keys.
        Fields that cannot be derived from the database are taken from
        ``extra_inputs``; absent keys default to zero so the formula's own
        division-by-zero guards emit ``None`` rather than raising.

        Parameters
        ----------
        business_type:
            Registry pack key (category/business type).
        agg_inputs:
            Numerical values fetched from ``CanonicalInsightRecord``.
        extra_inputs:
            Caller-supplied values for fields not available in the DB.

        Returns
        -------
        tuple[dict[str, float | None], dict[str, dict[str, Any]]]
            Raw metric values keyed by metric name, and per-metric validity
            metadata for the payload builder.
        """
        pack = _require_pack_or_raise(business_type)
        formula_inputs = _build_formula_inputs(pack, agg_inputs, extra_inputs)
        formula = pack.formula
        logger.debug("_compute invoking formula for business_type=%r", business_type)
        metrics = formula.calculate(formula_inputs)
        logger.debug("_compute metrics=%s", metrics)

        validity = _build_validity(pack, agg_inputs, extra_inputs, metrics)
        return metrics, validity

    # ------------------------------------------------------------------
    # Internal: precomputed metric backfill
    # ------------------------------------------------------------------

    def _backfill_from_precomputed(
        self,
        *,
        metrics: dict[str, float | None],
        validity: dict[str, dict[str, Any]],
        entity_name: str,
        period_start: datetime,
        period_end: datetime,
        pack: CategoryPack,
        db: Session,
    ) -> tuple[dict[str, float | None], dict[str, dict[str, Any]]]:
        """
        For any metric that the formula returned ``None``, attempt to fetch
        a pre-computed value directly from canonical insight records.

        Some datasets ship with already-derived metrics (e.g. ``churn_rate``,
        ``growth_rate``, ``ltv``) instead of the raw building blocks the
        formula needs.  This method fills the gaps so that the persisted
        KPI payload contains real values instead of blanket ``None``.
        """
        if not pack.precomputed_metrics:
            return metrics, validity

        none_metrics = [name for name, value in metrics.items() if value is None]
        precomputed_names = set(pack.precomputed_metrics)
        candidates = [m for m in none_metrics if m in precomputed_names]

        if not candidates:
            return metrics, validity

        agg = AggregationService(
            db,
            categories=category_aliases_for_business_type(pack.name),
        )
        precomputed = agg.get_precomputed_metrics(
            entity_name, period_start, period_end, candidates,
        )

        backfilled = 0
        for name, value in precomputed.items():
            if value is not None:
                metrics[name] = value
                # Replace validity error with provenance marker
                validity[name] = {
                    "is_valid": True,
                    "missing_dependencies": [],
                    "source": "precomputed_backfill",
                }
                backfilled += 1

        if backfilled:
            logger.info(
                "KPIOrchestrator._backfill_from_precomputed entity=%r: "
                "backfilled %d/%d metrics from precomputed canonical values",
                entity_name, backfilled, len(candidates),
            )

        return metrics, validity

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
        analytics_version: int | None = None,
        dataset_hash: str | None = None,
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
                analytics_version=analytics_version,
                dataset_hash=dataset_hash,
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
    active_customers: int | None
    lost_customers: int | None
    arpu: float | None
    current_revenue: float | None
    previous_revenue: float | None


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


def _require_pack_or_raise(business_type: str) -> CategoryPack:
    try:
        return require_category_pack(business_type)
    except CategoryRegistryError as exc:  # pragma: no cover
        raise KPIUnknownBusinessTypeError(
            f"Unknown business_type {business_type!r}. "
            f"Valid types: {list(supported_categories())}"
        ) from exc


def _resolve_source_value(
    source: str,
    inputs: _AggregatedInputs,
    extra: Mapping[str, Any],
    *,
    default: Any,
) -> Any:
    prefix, _, field_name = source.partition(".")
    prefix = prefix.strip().lower()
    key = field_name.strip()
    if prefix == "agg":
        value = getattr(inputs, key, None)
        if value is None:
            return _clone_default(default)
        return value
    if prefix == "extra":
        if key in extra:
            return extra[key]
        return _clone_default(default)
    raise ValueError(f"Unsupported binding source {source!r}")  # pragma: no cover


def _clone_default(value: Any) -> Any:
    return copy.deepcopy(value)


def _dependency_missing(
    rule: DependencyRule,
    inputs: _AggregatedInputs,
    extra: Mapping[str, Any],
) -> bool:
    value = _resolve_source_value(rule.source, inputs, extra, default=None)
    if rule.missing_when == "is_empty":
        if value is None:
            return True
        if isinstance(value, (str, bytes)):
            return not value
        if isinstance(value, (list, tuple, set, dict)):
            return len(value) == 0
        return False
    return value is None


def _dependency_label(source: str) -> str:
    _, _, label = source.partition(".")
    return label.strip() or source


def _build_formula_inputs(
    pack: CategoryPack,
    inputs: _AggregatedInputs,
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Build formula input dict from registry-defined bindings.
    """
    payload: dict[str, Any] = {}
    for key, binding in pack.formula_input_bindings.items():
        payload[key] = _resolve_source_value(binding.source, inputs, extra, default=binding.default)
    return payload


def _build_validity(
    pack: CategoryPack,
    inputs: _AggregatedInputs,
    extra: Mapping[str, Any],
    metrics: dict[str, float | None],
) -> dict[str, dict[str, Any]]:
    """
    Build per-metric validity metadata from registry-defined dependency rules.
    """
    validity: dict[str, dict[str, Any]] = {}

    for metric_name, dependencies in pack.validity_rules.items():
        missing = [
            _dependency_label(dep.source)
            for dep in dependencies
            if _dependency_missing(dep, inputs, extra)
        ]
        if missing:
            validity[metric_name] = {
                "is_valid": False,
                "missing_dependencies": missing,
                "status": "insufficient_data",
            }

    # For any metric the formula returned None that we haven't already
    # flagged, mark as invalid with a generic error.
    for name, value in metrics.items():
        if value is None and name not in validity:
            validity[name] = {
                "is_valid": False,
                "missing_dependencies": [],
                "status": "division_by_zero",
            }

    return validity


def _build_payload(
    metrics: dict[str, float | None],
    validity: dict[str, dict[str, Any]] | None = None,
    *,
    rate_metrics: frozenset[str] | None = None,
) -> dict[str, Any]:
    """
    Serialise formula output into the JSONB structure stored in ``computed_kpis``.

    Shape::

        {
            "mrr":         {"value": 12000.0, "unit": "currency", "error": null,
                            "is_valid": true, "missing_dependencies": []},
            "churn_rate":  {"value": 0.05,    "unit": "rate",     "error": null,
                            "is_valid": true, "missing_dependencies": []},
            "ltv":         {"value": null,    "unit": "currency",
                            "error": "insufficient_data", "is_valid": false,
                            "missing_dependencies": ["revenue", "churn_rate"]},
        }

    The ``"unit"`` is ``"rate"`` for metrics listed in ``rate_metrics``
    and ``"currency"`` for all others.

    Parameters
    ----------
    metrics:
        Raw metric values keyed by metric name.
    validity:
        Optional per-metric validity info.  Each entry may contain
        ``"is_valid"`` (bool), ``"missing_dependencies"`` (list[str]),
        and ``"status"`` (str).
    rate_metrics:
        Metrics that should be serialized as ``"unit": "rate"``.
    """
    val = validity or {}
    rates = rate_metrics or _DEFAULT_RATE_METRICS
    result: dict[str, Any] = {}
    for name, value in metrics.items():
        meta = val.get(name, {})
        is_valid = meta.get("is_valid", value is not None)
        missing_deps: list[str] = meta.get("missing_dependencies", [])
        status = meta.get("status")

        if not is_valid and value is None:
            error = status or "insufficient_data"
        elif value is None:
            error = "division_by_zero"
        else:
            error = None

        source = meta.get("source", "formula")
        result[name] = {
            "value": value,
            "unit": "rate" if name in rates else "currency",
            "error": error,
            "is_valid": is_valid,
            "missing_dependencies": missing_deps,
            "source": source,
        }
    return result
