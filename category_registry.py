"""
category_registry.py

Simple category -> processing strategy registry.
"""

from __future__ import annotations

_DEFAULT_PROCESSING_STRATEGY = "generic_timeseries"

CATEGORY_TO_PROCESSING_STRATEGY: dict[str, str] = {
    "saas": "saas",
    "ecommerce": "ecommerce",
    "agency": "agency",
    "financial_market": "generic_timeseries",
    "generic_timeseries": "generic_timeseries",
}


def _normalize(value: str | None) -> str:
    return str(value or "").strip().lower()


def get_processing_strategy(category: str | None) -> str | None:
    """
    Resolve a category into a processing strategy.

    Unknown non-empty categories default to generic_timeseries so router logic
    does not need category-specific branches.
    """
    normalized = _normalize(category)
    if not normalized:
        return None
    return CATEGORY_TO_PROCESSING_STRATEGY.get(
        normalized,
        _DEFAULT_PROCESSING_STRATEGY,
    )


def register_category(category: str, processing_strategy: str) -> None:
    """Register or update a category mapping at runtime."""
    normalized_category = _normalize(category)
    normalized_strategy = _normalize(processing_strategy)
    if not normalized_category:
        raise ValueError("category must be a non-empty string")
    if not normalized_strategy:
        raise ValueError("processing_strategy must be a non-empty string")
    CATEGORY_TO_PROCESSING_STRATEGY[normalized_category] = normalized_strategy


def supported_categories() -> tuple[str, ...]:
    """Return configured categories in stable order."""
    return tuple(sorted(CATEGORY_TO_PROCESSING_STRATEGY))

