"""
app/services/canonical_validation.py

Pre-aggregation canonical data integrity checks for KPI computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import Text, and_, cast, or_, select
from sqlalchemy.orm import Session

from app.services.category_registry import get_category_pack
from app.services.kpi_canonical_schema import (
    category_aliases_for_business_type,
    metric_aliases_for_business_type,
    required_metric_keys_for_business_type,
)
from db.models.canonical_insight_record import CanonicalInsightRecord


@dataclass(frozen=True)
class CanonicalValidationResult:
    is_valid: bool
    missing_metrics: list[str]
    error_payload: dict[str, Any] | None = None


def validate_canonical_inputs_for_kpi(
    *,
    db: Session,
    entity_name: str,
    business_type: str,
    period_start: datetime,
    period_end: datetime,
) -> CanonicalValidationResult:
    """
    Validate canonical records before running aggregation/formula computation.

    Checks:
    - minimum row count > 0 for entity/category/period
    - required metrics exist for business type
    - required metrics have non-null metric_value
    """
    required_metric_keys = _required_metrics_for_business_type(business_type)
    metric_aliases = metric_aliases_for_business_type(business_type)
    category_aliases = category_aliases_for_business_type(business_type)
    all_metric_aliases = sorted(
        {
            alias
            for metric_key in required_metric_keys
            for alias in metric_aliases.get(metric_key, (metric_key,))
        }
    )

    base_predicate = _base_predicate(
        entity_name=entity_name,
        categories=category_aliases,
        period_start=period_start,
        period_end=period_end,
    )
    non_null_metric_value = and_(
        CanonicalInsightRecord.metric_value.is_not(None),
        cast(CanonicalInsightRecord.metric_value, Text) != "null",
    )
    null_metric_value = or_(
        CanonicalInsightRecord.metric_value.is_(None),
        cast(CanonicalInsightRecord.metric_value, Text) == "null",
    )

    total_rows = db.scalar(
        select(CanonicalInsightRecord.id).where(base_predicate).limit(1)
    )
    if total_rows is None:
        missing = sorted(required_metric_keys)
        return CanonicalValidationResult(
            is_valid=False,
            missing_metrics=missing,
            error_payload={
                "error_type": "canonical_validation_failed",
                "missing_metrics": missing,
            },
        )

    available_metric_names = set(
        db.scalars(
            select(CanonicalInsightRecord.metric_name)
            .where(
                base_predicate,
                CanonicalInsightRecord.metric_name.in_(all_metric_aliases),
                non_null_metric_value,
            )
            .group_by(CanonicalInsightRecord.metric_name)
        ).all()
    )

    null_value_metric_names = set(
        db.scalars(
            select(CanonicalInsightRecord.metric_name)
            .where(
                base_predicate,
                CanonicalInsightRecord.metric_name.in_(all_metric_aliases),
                null_metric_value,
            )
            .group_by(CanonicalInsightRecord.metric_name)
        ).all()
    )

    missing_metrics: list[str] = []
    for metric_key in required_metric_keys:
        aliases = set(metric_aliases.get(metric_key, (metric_key,)))
        has_non_null_value = bool(aliases & available_metric_names)
        # A metric is only missing if there are NO non-null rows at all.
        # Having some null rows alongside valid rows is acceptable — the
        # aggregation layer already handles None gracefully.
        if not has_non_null_value:
            missing_metrics.append(metric_key)

    if missing_metrics:
        # Before failing, check if precomputed output metrics exist that
        # cover the gap.  E.g. if active_customer_count is missing but the
        # dataset ships churn_rate and ltv directly, the formula can be
        # bypassed for those outputs via the precomputed passthrough.
        pack = get_category_pack(business_type)
        precomputed_names: set[str] = set()
        if pack and pack.precomputed_metrics:
            precomputed_names = set(pack.precomputed_metrics)

        if precomputed_names:
            # Check which precomputed metrics actually exist in canonical records
            available_precomputed = set(
                db.scalars(
                    select(CanonicalInsightRecord.metric_name)
                    .where(
                        base_predicate,
                        CanonicalInsightRecord.metric_name.in_(tuple(precomputed_names)),
                        non_null_metric_value,
                    )
                    .group_by(CanonicalInsightRecord.metric_name)
                ).all()
            )
            if available_precomputed:
                # Precomputed output metrics exist — allow validation to pass.
                # The orchestrator will use the precomputed passthrough for
                # metrics whose raw building blocks are missing.
                return CanonicalValidationResult(
                    is_valid=True,
                    missing_metrics=missing_metrics,
                    error_payload=None,
                )

        return CanonicalValidationResult(
            is_valid=False,
            missing_metrics=missing_metrics,
            error_payload={
                "error_type": "canonical_validation_failed",
                "missing_metrics": missing_metrics,
            },
        )

    return CanonicalValidationResult(
        is_valid=True,
        missing_metrics=[],
        error_payload=None,
    )


def _required_metrics_for_business_type(business_type: str) -> tuple[str, ...]:
    return required_metric_keys_for_business_type(business_type)


def _base_predicate(
    *,
    entity_name: str,
    categories: tuple[str, ...],
    period_start: datetime,
    period_end: datetime,
):
    return and_(
        CanonicalInsightRecord.entity_name == entity_name,
        CanonicalInsightRecord.category.in_(categories),
        CanonicalInsightRecord.timestamp >= period_start,
        CanonicalInsightRecord.timestamp < period_end,
    )
