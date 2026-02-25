from __future__ import annotations

from agent.graph import derive_pipeline_status
from agent.nodes.business_router import route_by_business_type
from agent.nodes.node_result import skipped, success
from app.services.category_registry import get_processing_strategy, require_category_pack


def test_general_timeseries_pack_exists() -> None:
    pack = require_category_pack("general_timeseries")
    assert pack.name == "general_timeseries"
    assert "timeseries_value" in pack.metric_aliases["recurring_revenue"]


def test_generic_timeseries_alias_maps_to_general_timeseries() -> None:
    assert get_processing_strategy("generic_timeseries") == "general_timeseries"


def test_business_router_routes_general_timeseries_to_generic_kpi_fetch() -> None:
    route = route_by_business_type({"business_type": "general_timeseries"})
    assert route == "kpi_fetch"


def test_business_router_unknown_falls_back_to_generic_kpi_fetch() -> None:
    route = route_by_business_type({"business_type": "unrecognized_category"})
    assert route == "kpi_fetch"


def test_pipeline_status_for_general_timeseries_uses_kpi_data_requirement() -> None:
    state = {
        "business_type": "general_timeseries",
        "kpi_data": success({"records": [{"computed_kpis": {"timeseries_value": {"value": 10.0}}}]}),
        "risk_data": skipped("unsupported_business_type"),
        "forecast_data": None,
        "root_cause": None,
    }
    assert derive_pipeline_status(state) == "partial"
