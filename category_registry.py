"""
Compatibility facade for category dispatch helpers.

Use ``app.services.category_registry`` directly for new code.
"""

from __future__ import annotations

from app.services.category_registry import (
    get_processing_strategy,
    supported_categories,
)


def register_category(category: str, processing_strategy: str) -> None:
    """
    Backward-compatible API stub.

    Runtime registration is no longer supported; categories are loaded from
    YAML packs in ``config/categories``.
    """
    raise RuntimeError(
        "register_category() is no longer supported. "
        "Add a YAML pack under config/categories instead."
    )
