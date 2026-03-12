"""
app/api/routers package marker with lazy router exports.

Avoids importing all routers eagerly on package import. This keeps optional
dependencies isolated and prevents unrelated import failures during targeted
module loads.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_ROUTER_IMPORTS: dict[str, tuple[str, str]] = {
    "analyze_router": ("app.api.routers.analyze_router", "router"),
    "bi_export_router": ("app.api.routers.bi_export_router", "router"),
    "business_intelligence_router": ("app.api.routers.business_intelligence_router", "router"),
    "client_router": ("app.api.routers.client_router", "router"),
    "competitor_scraping_router": ("app.api.routers.competitor_scraping", "router"),
    "csv_ingestion_router": ("app.api.routers.csv_ingestion", "router"),
    "dashboard_router": ("app.api.routers.dashboard_router", "router"),
    "decision_engine_router": ("app.api.routers.decision_engine_router", "router"),
    "external_ingestion_router": ("app.api.routers.external_ingestion", "router"),
    "ingestion_orchestrator_router": ("app.api.routers.ingestion_orchestrator", "router"),
    "kpi_router": ("app.api.routers.kpi_router", "router"),
}

__all__ = list(_ROUTER_IMPORTS.keys())


def __getattr__(name: str) -> Any:
    spec = _ROUTER_IMPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = spec
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

