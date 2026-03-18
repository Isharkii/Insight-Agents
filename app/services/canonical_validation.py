"""
app/services/canonical_validation.py

Pre-aggregation canonical data integrity checks for KPI computation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import Text, and_, cast, func, or_, select
from sqlalchemy.orm import Session

from app.services.category_registry import get_category_pack
from app.services.kpi_canonical_schema import (
    category_aliases_for_business_type,
    metric_aliases_for_business_type,
    required_metric_keys_for_business_type,
)
from db.models.canonical_insight_record import CanonicalInsightRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CanonicalValidationResult:
    is_valid: bool
    missing_metrics: list[str]
    error_payload: dict[str, Any] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


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

    Uses case-insensitive matching via ``lower()`` to prevent casing
    mismatches between ingested data and canonical alias definitions.
    """
    required_metric_keys = _required_metrics_for_business_type(business_type)
    metric_aliases = metric_aliases_for_business_type(business_type)
    category_aliases = category_aliases_for_business_type(business_type)
    all_metric_aliases = sorted(
        {
            alias.lower()
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

    # Use lower() for case-insensitive metric_name matching
    metric_name_lower = func.lower(CanonicalInsightRecord.metric_name)

    total_rows = db.scalar(
        select(CanonicalInsightRecord.id).where(base_predicate).limit(1)
    )

    # Debug: fetch ALL distinct metric_names present for this entity/category/period
    all_db_metrics: set[str] = set()
    if total_rows is not None:
        all_db_metrics = set(
            db.scalars(
                select(CanonicalInsightRecord.metric_name)
                .where(base_predicate)
                .group_by(CanonicalInsightRecord.metric_name)
            ).all()
        )

    if total_rows is None:
        missing = sorted(required_metric_keys)
        logger.warning(
            "Canonical validation: ZERO rows found for entity=%r "
            "categories=%s period=[%s, %s). "
            "All required metrics reported as missing: %s",
            entity_name,
            category_aliases,
            period_start.isoformat(),
            period_end.isoformat(),
            missing,
        )
        return CanonicalValidationResult(
            is_valid=False,
            missing_metrics=missing,
            error_payload={
                "error_type": "canonical_validation_failed",
                "missing_metrics": missing,
                "reason": "no_rows_found",
            },
            diagnostics={
                "entity_name": entity_name,
                "categories_queried": list(category_aliases),
                "period_start": period_start.isoformat(),
                "period_end": period_end.isoformat(),
                "total_rows_found": 0,
                "db_metric_names": sorted(all_db_metrics),
                "required_aliases_queried": all_metric_aliases,
            },
        )

    logger.debug(
        "Canonical validation: %d+ rows found for entity=%r. "
        "DB metric_names=%s, required aliases=%s",
        1,
        entity_name,
        sorted(all_db_metrics),
        all_metric_aliases,
    )

    # Case-insensitive query for available metrics
    available_metric_names = set(
        db.scalars(
            select(metric_name_lower)
            .where(
                base_predicate,
                metric_name_lower.in_(all_metric_aliases),
                non_null_metric_value,
            )
            .group_by(metric_name_lower)
        ).all()
    )

    null_value_metric_names = set(
        db.scalars(
            select(metric_name_lower)
            .where(
                base_predicate,
                metric_name_lower.in_(all_metric_aliases),
                null_metric_value,
            )
            .group_by(metric_name_lower)
        ).all()
    )

    missing_metrics: list[str] = []
    for metric_key in required_metric_keys:
        aliases = {a.lower() for a in metric_aliases.get(metric_key, (metric_key,))}
        has_non_null_value = bool(aliases & available_metric_names)
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
            precomputed_names = {p.lower() for p in pack.precomputed_metrics}

        available_precomputed: set[str] = set()
        if precomputed_names:
            available_precomputed = set(
                db.scalars(
                    select(metric_name_lower)
                    .where(
                        base_predicate,
                        metric_name_lower.in_(tuple(precomputed_names)),
                        non_null_metric_value,
                    )
                    .group_by(metric_name_lower)
                ).all()
            )
            if available_precomputed:
                logger.info(
                    "Canonical validation: missing_metrics=%s but precomputed "
                    "passthrough active (found: %s). Validation PASSED.",
                    missing_metrics,
                    sorted(available_precomputed),
                )
                return CanonicalValidationResult(
                    is_valid=True,
                    missing_metrics=missing_metrics,
                    error_payload=None,
                    diagnostics={
                        "entity_name": entity_name,
                        "categories_queried": list(category_aliases),
                        "period_start": period_start.isoformat(),
                        "period_end": period_end.isoformat(),
                        "db_metric_names": sorted(all_db_metrics),
                        "available_required": sorted(available_metric_names),
                        "null_only_metrics": sorted(null_value_metric_names - available_metric_names),
                        "precomputed_found": sorted(available_precomputed),
                        "precomputed_expected": sorted(precomputed_names),
                        "passthrough_active": True,
                    },
                )

        logger.warning(
            "Canonical validation FAILED: entity=%r business_type=%r "
            "missing_metrics=%s. DB has metric_names=%s, "
            "available_required=%s, null_only=%s, "
            "precomputed_expected=%s precomputed_found=%s",
            entity_name,
            business_type,
            missing_metrics,
            sorted(all_db_metrics),
            sorted(available_metric_names),
            sorted(null_value_metric_names - available_metric_names),
            sorted(precomputed_names),
            sorted(available_precomputed),
        )
        return CanonicalValidationResult(
            is_valid=False,
            missing_metrics=missing_metrics,
            error_payload={
                "error_type": "canonical_validation_failed",
                "missing_metrics": missing_metrics,
            },
            diagnostics={
                "entity_name": entity_name,
                "categories_queried": list(category_aliases),
                "period_start": period_start.isoformat(),
                "period_end": period_end.isoformat(),
                "db_metric_names": sorted(all_db_metrics),
                "available_required": sorted(available_metric_names),
                "null_only_metrics": sorted(null_value_metric_names - available_metric_names),
                "precomputed_expected": sorted(precomputed_names),
                "precomputed_found": sorted(available_precomputed),
                "passthrough_active": False,
            },
        )

    logger.debug(
        "Canonical validation PASSED: entity=%r all required metrics present.",
        entity_name,
    )
    return CanonicalValidationResult(
        is_valid=True,
        missing_metrics=[],
        error_payload=None,
        diagnostics={
            "entity_name": entity_name,
            "db_metric_names": sorted(all_db_metrics),
            "available_required": sorted(available_metric_names),
        },
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
