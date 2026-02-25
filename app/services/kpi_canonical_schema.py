"""
app/services/kpi_canonical_schema.py

Shared canonical KPI alias rules used by validation, aggregation, and routing.
"""

from __future__ import annotations

from typing import Final

from app.services.category_registry import get_category_pack, supported_categories

KPI_REQUIRED_METRIC_KEYS: Final[tuple[str, str, str]] = (
    "recurring_revenue",
    "active_customer_count",
    "churned_customer_count",
)

_DEFAULT_METRIC_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "recurring_revenue": ("recurring_revenue",),
    "active_customer_count": ("active_customer_count",),
    "churned_customer_count": ("churned_customer_count",),
}


def _normalize(value: str | None) -> str:
    return str(value or "").strip().lower()


def metric_aliases_for_business_type(
    business_type: str,
) -> dict[str, tuple[str, ...]]:
    """Return metric aliases keyed by canonical KPI metric key."""
    pack = get_category_pack(_normalize(business_type))
    if pack is None:
        return dict(_DEFAULT_METRIC_ALIASES)
    return dict(pack.metric_aliases)


def category_aliases_for_business_type(business_type: str) -> tuple[str, ...]:
    """Return canonical category aliases accepted for KPI source rows."""
    pack = get_category_pack(_normalize(business_type))
    if pack is None:
        return ("sales",)
    return pack.category_aliases


def required_metric_keys_for_business_type(business_type: str) -> tuple[str, ...]:
    """Return required canonical metric keys for category validation."""
    pack = get_category_pack(_normalize(business_type))
    if pack is None:
        return KPI_REQUIRED_METRIC_KEYS
    return pack.required_inputs


def infer_analytics_strategy_from_categories(categories: list[str] | tuple[str, ...]) -> str | None:
    """
    Infer KPI analytics strategy from dataset categories.

    Returns a strategy only when exactly one supported KPI category is present.
    """
    supported = set(supported_categories())
    discovered: list[str] = []
    for raw in categories:
        category = _normalize(raw)
        if category in supported and category not in discovered:
            discovered.append(category)

    if len(discovered) == 1:
        return discovered[0]
    return None
