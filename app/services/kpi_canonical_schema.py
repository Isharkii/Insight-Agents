"""
app/services/kpi_canonical_schema.py

Shared canonical KPI alias rules used by validation, aggregation, and routing.
"""

from __future__ import annotations

from typing import Final

KPI_SUPPORTED_ANALYTICS_STRATEGIES: Final[frozenset[str]] = frozenset(
    {"saas", "ecommerce", "agency"}
)

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

_METRIC_ALIASES_BY_BUSINESS_TYPE: Final[dict[str, dict[str, tuple[str, ...]]]] = {
    "saas": {
        "recurring_revenue": ("recurring_revenue", "mrr"),
        "active_customer_count": ("active_customer_count", "active_customers"),
        "churned_customer_count": (
            "churned_customer_count",
            "churned_subscriptions",
            "churned_customers",
        ),
    },
    "ecommerce": {
        "recurring_revenue": ("recurring_revenue", "revenue", "gmv"),
        "active_customer_count": ("active_customer_count", "active_customers"),
        "churned_customer_count": (
            "churned_customer_count",
            "churned_customers",
            "lost_customers",
        ),
    },
    "agency": {
        "recurring_revenue": ("recurring_revenue", "total_revenue"),
        "active_customer_count": ("active_customer_count", "active_clients"),
        "churned_customer_count": (
            "churned_customer_count",
            "churned_clients",
        ),
    },
}

_CATEGORY_ALIASES_BY_BUSINESS_TYPE: Final[dict[str, tuple[str, ...]]] = {
    "saas": ("sales", "saas"),
    "ecommerce": ("sales", "ecommerce"),
    "agency": ("sales", "agency"),
}


def _normalize(value: str | None) -> str:
    return str(value or "").strip().lower()


def metric_aliases_for_business_type(
    business_type: str,
) -> dict[str, tuple[str, ...]]:
    """Return metric aliases keyed by canonical KPI metric key."""
    normalized = _normalize(business_type)
    aliases = _METRIC_ALIASES_BY_BUSINESS_TYPE.get(normalized, _DEFAULT_METRIC_ALIASES)
    return dict(aliases)


def category_aliases_for_business_type(business_type: str) -> tuple[str, ...]:
    """Return canonical category aliases accepted for KPI source rows."""
    normalized = _normalize(business_type)
    return _CATEGORY_ALIASES_BY_BUSINESS_TYPE.get(normalized, ("sales",))


def infer_analytics_strategy_from_categories(categories: list[str] | tuple[str, ...]) -> str | None:
    """
    Infer KPI analytics strategy from dataset categories.

    Returns a strategy only when exactly one supported KPI category is present.
    """
    discovered: list[str] = []
    for raw in categories:
        category = _normalize(raw)
        if category in KPI_SUPPORTED_ANALYTICS_STRATEGIES and category not in discovered:
            discovered.append(category)

    if len(discovered) == 1:
        return discovered[0]
    return None
