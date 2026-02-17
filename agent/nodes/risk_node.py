"""
agent/nodes/risk_node.py

Risk Node: extracts signal values from state.kpi_data and state.forecast_data,
calls RiskOrchestrator, and stores the result in state.risk_data.

No scoring math, no schema changes.
"""

from __future__ import annotations

from typing import Any

from agent.state import AgentState
from risk.orchestrator import RiskOrchestrator
from db.session import SessionLocal


# ---------------------------------------------------------------------------
# Signal extractors
# ---------------------------------------------------------------------------

def _kpi_signals(kpi_data: dict | None) -> dict[str, float]:
    """
    Pull the three KPI delta signals RiskOrchestrator expects.

    Searches the first record in kpi_data["records"] that contains the key
    inside its ``computed_kpis`` payload.  Defaults to 0.0 for any missing
    signal so the orchestrator always receives a complete dict.
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
                signals[key] = float(computed[key].get("value", computed[key])
                                     if isinstance(computed[key], dict)
                                     else computed[key])
    return signals


def _forecast_signals(forecast_data: dict | None) -> dict[str, float]:
    """
    Pull the three forecast signals RiskOrchestrator expects.

    Checks each metric's ``forecast_data`` payload for the signal key.
    Defaults to 0.0 for anything absent.
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


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def risk_node(state: AgentState) -> AgentState:
    """
    LangGraph node: generate a risk score for the entity.

    Reads:
        state["entity_name"]    — entity being scored.
        state["kpi_data"]       — output of kpi_fetch_node.
        state["forecast_data"]  — output of forecast_fetch_node.

    Writes:
        state["risk_data"] — dict with keys:
            "entity_name" : str
            "risk_score"  : int   (0–100)
            "risk_level"  : str   ("low" | "moderate" | "high" | "critical")
            "error"       : str   (present only on failure)
    """
    entity_name: str = state.get("entity_name") or "unknown"
    kpi_data: dict | None = state.get("kpi_data")
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
