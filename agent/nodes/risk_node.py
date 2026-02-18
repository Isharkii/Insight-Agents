"""
agent/nodes/risk_node.py

Risk Node: extracts signal values from one business-specific KPI payload and
state.forecast_data, calls RiskOrchestrator, and stores the result in
state.risk_data.

No scoring math, no schema changes.
"""

from __future__ import annotations

from typing import Any

from agent.state import AgentState
from db.session import SessionLocal
from risk.orchestrator import RiskOrchestrator

_KPI_KEY_BY_BUSINESS_TYPE: dict[str, str] = {
    "saas": "saas_kpi_data",
    "software": "saas_kpi_data",
    "ecommerce": "ecommerce_kpi_data",
    "retail": "ecommerce_kpi_data",
    "food_service": "ecommerce_kpi_data",
    "agency": "agency_kpi_data",
    "marketing": "agency_kpi_data",
    "consulting": "agency_kpi_data",
}


# ---------------------------------------------------------------------------
# Signal extractors
# ---------------------------------------------------------------------------

def _kpi_signals(kpi_data: dict | None) -> dict[str, float]:
    """
    Pull the three KPI delta signals RiskOrchestrator expects.

    Searches records in kpi_data["records"] for the required keys inside
    each record's computed_kpis payload. Missing values default to 0.0.
    """
    signals = {
        "revenue_growth_delta": 0.0,
        "churn_delta": 0.0,
        "conversion_delta": 0.0,
    }
    if not kpi_data:
        return signals

    for record in kpi_data.get("records", []):
        computed: dict[str, Any] = record.get("computed_kpis", {})
        for key in signals:
            if key in computed and signals[key] == 0.0:
                value = computed[key]
                if isinstance(value, dict):
                    value = value.get("value", value)
                signals[key] = float(value)
    return signals


def _forecast_signals(forecast_data: dict | None) -> dict[str, float]:
    """
    Pull the three forecast signals RiskOrchestrator expects.

    Checks each metric's forecast_data payload for required keys. Missing
    values default to 0.0.
    """
    signals = {
        "deviation_percentage": 0.0,
        "slope": 0.0,
        "churn_acceleration": 0.0,
    }
    if not forecast_data:
        return signals

    for _metric, row in forecast_data.get("forecasts", {}).items():
        if row is None:
            continue
        payload: dict[str, Any] = row.get("forecast_data", {})
        for key in signals:
            if key in payload and signals[key] == 0.0:
                signals[key] = float(payload[key])
    return signals


def _kpi_data_for_business_type(state: AgentState) -> dict | None:
    """
    Select exactly one KPI payload based on state.business_type.

    No cross-payload merge is performed.
    """
    business_type = str(state.get("business_type") or "").lower()
    kpi_key = _KPI_KEY_BY_BUSINESS_TYPE.get(business_type)
    if kpi_key is not None:
        return state.get(kpi_key)
    return state.get("kpi_data")


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def risk_node(state: AgentState) -> AgentState:
    """
    LangGraph node: generate a risk score for the entity.

    Reads:
        state["entity_name"]   - entity being scored.
        state["business_type"] - selects one KPI payload only.
        state["forecast_data"] - output of forecast_fetch_node.

    Writes:
        state["risk_data"] - dict with keys:
            "entity_name" : str
            "risk_score"  : int (0-100)
            "risk_level"  : str ("low" | "moderate" | "high" | "critical")
            "error"       : str (present only on failure)
    """
    entity_name: str = state.get("entity_name") or "unknown"
    kpi_data: dict | None = _kpi_data_for_business_type(state)
    forecast_data: dict | None = state.get("forecast_data")

    kpi_signals = _kpi_signals(kpi_data)
    forecast_signals = _forecast_signals(forecast_data)

    try:
        with SessionLocal() as session:
            orchestrator = RiskOrchestrator(session)
            result = orchestrator.generate_risk_score(
                entity_name=entity_name,
                kpi_data=kpi_signals,
                forecast_data=forecast_signals,
            )
            session.commit()

        risk_data: dict[str, Any] = result

    except Exception as exc:  # noqa: BLE001
        risk_data = {
            "entity_name": entity_name,
            "risk_score": None,
            "risk_level": None,
            "error": str(exc),
        }

    return {**state, "risk_data": risk_data}
