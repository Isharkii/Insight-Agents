"""
agent/nodes/forecast_node.py

Forecast Fetch Node: retrieves the latest persisted forecast for each
relevant metric of the entity named in state.

No forecasting math, no risk logic, no DB schema changes.
"""

from __future__ import annotations

from typing import Any

from agent.nodes.node_result import failed, skipped, success
from agent.state import AgentState
from db.session import SessionLocal
from forecast.repository import ForecastRepository

# ---------------------------------------------------------------------------
# Metric sets per business type
# ---------------------------------------------------------------------------

_DEFAULT_METRICS: list[str] = ["revenue", "growth_rate", "sales"]

_METRICS_BY_BUSINESS_TYPE: dict[str, list[str]] = {
    "saas": ["mrr", "churn_rate", "ltv", "growth_rate"],
    "ecommerce": [
        "revenue",
        "aov",
        "conversion_rate",
        "cac",
        "purchase_frequency",
        "ltv",
        "growth_rate",
    ],
    "agency": [
        "total_revenue",
        "client_churn",
        "utilization_rate",
        "revenue_per_employee",
        "client_ltv",
    ],
}


def _metrics_for(business_type: str) -> list[str]:
    return _METRICS_BY_BUSINESS_TYPE.get(business_type, _DEFAULT_METRICS)


def _serialize_row(row: Any) -> dict[str, Any]:
    """Convert a ForecastMetric ORM row to a plain JSON-safe dict."""
    return {
        "entity_name": row.entity_name,
        "metric_name": row.metric_name,
        "period_end": row.period_end.isoformat(),
        "forecast_data": row.forecast_data,
        "created_at": row.created_at.isoformat(),
    }


def forecast_fetch_node(state: AgentState) -> AgentState:
    """
    LangGraph node: fetch the latest forecast records from the repository.

    Reads:
        state["entity_name"]    — entity whose forecasts are fetched.
        state["business_type"]  — determines which metrics to look up.

    Writes:
        state["forecast_data"] — dict with keys:
            "forecasts"      : dict[metric_name, serialised row or None]
            "fetched_for"    : entity_name used for the query
            "metrics_queried": list of metric names attempted
            "error"          : present only on failure
    """
    entity_name: str = state.get("entity_name") or ""
    business_type: str = state.get("business_type") or "general"
    metrics = _metrics_for(business_type)

    try:
        forecasts: dict[str, Any] = {}

        with SessionLocal() as session:
            repo = ForecastRepository(session)
            for metric in metrics:
                row = repo.get_latest_forecast(
                    entity_name=entity_name,
                    metric_name=metric,
                )
                forecasts[metric] = _serialize_row(row) if row else None

        payload: dict[str, Any] = {
            "forecasts": forecasts,
            "fetched_for": entity_name,
            "metrics_queried": metrics,
        }
        has_any_forecast = any(row is not None for row in forecasts.values())
        if has_any_forecast:
            forecast_data = success(payload)
        else:
            forecast_data = skipped("no_forecast_records", payload)

    except Exception as exc:  # noqa: BLE001
        forecast_data = failed(
            str(exc),
            {
            "fetched_for": entity_name,
            "metrics_queried": metrics,
            },
        )

    return {**state, "forecast_data": forecast_data}
