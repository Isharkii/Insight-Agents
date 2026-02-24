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
from agent.nodes.node_result import failed, payload_of, skipped, status_of, success
from agent.signal_normalizer import normalize_forecast_signals, normalize_kpi_signals
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
        return {}

    payload = payload_of(state.get(kpi_key))
    return payload if isinstance(payload, dict) else {}


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
    business_type: str = str(state.get("business_type") or "").lower()
    kpi_payload: dict = _kpi_data_for_business_type(state)
    forecast_state = state.get("forecast_data")
    forecast_payload: dict = payload_of(forecast_state) or {}
    forecast_status = status_of(forecast_state)

    if business_type not in _KPI_KEY_BY_BUSINESS_TYPE:
        risk_data = skipped(
            "unsupported_business_type",
            {"business_type": business_type},
        )
        return {**state, "risk_data": risk_data}

    if not kpi_payload:
        risk_data = skipped(
            "kpi_unavailable",
            {"business_type": business_type},
        )
        return {**state, "risk_data": risk_data}

    try:
        kpi_signals = normalize_kpi_signals(kpi_payload)
    except Exception as exc:  # noqa: BLE001
        risk_data = failed(
            f"kpi_signal_normalization_failed: {exc}",
            {"business_type": business_type},
        )
        return {**state, "risk_data": risk_data}

    forecast_signals: dict[str, float] = {}
    forecast_context: dict[str, Any]
    if forecast_status == "success" and forecast_payload:
        try:
            forecast_signals = normalize_forecast_signals(forecast_payload)
            forecast_context = {
                "status": "ok",
                "forecast_available": True,
                **forecast_signals,
            }
        except Exception:
            forecast_context = {
                "status": "insufficient_data",
                "forecast_available": False,
            }
    else:
        forecast_context = {
            "status": "insufficient_data",
            "forecast_available": False,
        }

    try:
        with SessionLocal() as session:
            orchestrator = RiskOrchestrator(session)
            result = orchestrator.generate_risk_score(
                entity_name=entity_name,
                kpi_data=kpi_signals,
                forecast_data=forecast_context,
            )
            session.commit()

        payload: dict[str, Any] = {
            **result,
            "forecast_available": bool(forecast_context.get("forecast_available")),
        }
        risk_data = success(payload)

    except Exception as exc:  # noqa: BLE001
        risk_data = failed(str(exc), {"entity_name": entity_name})

    return {**state, "risk_data": risk_data}
