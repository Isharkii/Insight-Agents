"""
agent/nodes/risk_node.py

Risk Node: normalizes signals from one business-specific KPI payload and
state.forecast_data, calls RiskOrchestrator, and stores the result in
state.risk_data.

No scoring math, no schema changes.
"""

from __future__ import annotations

from typing import Any

from agent.state import AgentState
from agent.signal_normalizer import normalize_signals
from db.session import SessionLocal
from risk.orchestrator import RiskOrchestrator

_KPI_KEY_BY_BUSINESS_TYPE: dict[str, str] = {
    "saas": "saas_kpi_data",
    "ecommerce": "ecommerce_kpi_data",
    "agency": "agency_kpi_data",
}


def _kpi_data_for_business_type(state: AgentState) -> dict:
    """
    Select exactly one KPI payload based on state.business_type.

    No cross-payload merge is performed.
    """
    business_type = str(state.get("business_type") or "").lower()
    kpi_key = _KPI_KEY_BY_BUSINESS_TYPE.get(business_type)
    if kpi_key is None:
        supported = ", ".join(sorted(_KPI_KEY_BY_BUSINESS_TYPE))
        raise ValueError(
            f"Unsupported business_type '{business_type}'. "
            f"Supported values: {supported}."
        )

    payload = state.get(kpi_key)
    if not isinstance(payload, dict):
        raise ValueError(f"Missing or invalid state['{kpi_key}'] payload.")
    return payload


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

    Raises:
        ValueError: If signal normalization fails.
    """
    entity_name: str = state.get("entity_name") or "unknown"
    business_type: str = str(state.get("business_type") or "").lower()
    kpi_payload: dict = _kpi_data_for_business_type(state)
    forecast_payload: dict = state.get("forecast_data") or {}

    try:
        flat_signals = normalize_signals(
            kpi_payload=kpi_payload,
            forecast_payload=forecast_payload,
        )
    except ValueError as exc:
        raise ValueError(
            "Signal normalization failed "
            f"for entity='{entity_name}', business_type='{business_type}': {exc}"
        ) from exc

    try:
        with SessionLocal() as session:
            orchestrator = RiskOrchestrator(session)
            result = orchestrator.generate_risk_score(
                entity_name=entity_name,
                kpi_data=flat_signals,
                forecast_data=flat_signals,
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
