"""
agent/nodes/risk_node.py

Risk Node: normalizes signals from one business-specific KPI payload and
state.forecast_data, calls RiskOrchestrator, and stores the result in
state.risk_data.

No scoring math, no schema changes.
"""

from __future__ import annotations

from typing import Any, Mapping

from agent.state import AgentState
from agent.nodes.node_result import (
    failed, insufficient_data, payload_of, skipped, status_of, success,
    warnings_of, confidence_of,
)
from agent.signal_normalizer import (
    normalize_forecast_signals,
    normalize_kpi_signals,
)
from agent.signal_integrity import UnifiedSignalIntegrity
from db.session import SessionLocal
from risk.orchestrator import RiskOrchestrator

_KPI_KEY_BY_BUSINESS_TYPE: dict[str, str] = {
    "saas": "saas_kpi_data",
    "ecommerce": "ecommerce_kpi_data",
    "agency": "agency_kpi_data",
    "general_timeseries": "kpi_data",
    "generic_timeseries": "kpi_data",
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


def _segmentation_or_node_payload(
    state: AgentState,
    *,
    segmentation_key: str,
    node_key: str,
) -> dict[str, Any]:
    segmentation = payload_of(state.get("segmentation")) or {}
    if isinstance(segmentation, dict):
        candidate = segmentation.get(segmentation_key)
        if isinstance(candidate, dict):
            return candidate
    candidate = payload_of(state.get(node_key))
    return candidate if isinstance(candidate, dict) else {}


def _cohort_signal_snapshot(state: AgentState) -> dict[str, Any]:
    cohort = _segmentation_or_node_payload(
        state,
        segmentation_key="cohort_analytics",
        node_key="cohort_data",
    )
    if not cohort:
        return {
            "available": False,
            "status": "missing",
            "confidence_score": 1.0,
            "warnings": [],
            "churn_acceleration": None,
            "risk_hint": "low",
        }

    signals = cohort.get("signals")
    if not isinstance(signals, dict):
        signals = {}

    confidence_raw = cohort.get("confidence_score")
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 1.0
    confidence = max(0.0, min(1.0, confidence))

    churn_acceleration = signals.get("churn_acceleration")
    try:
        churn_acceleration = float(churn_acceleration) if churn_acceleration is not None else None
    except (TypeError, ValueError):
        churn_acceleration = None

    warnings = cohort.get("warnings")
    if not isinstance(warnings, list):
        warnings = []

    return {
        "available": churn_acceleration is not None,
        "status": str(cohort.get("status") or "success").strip().lower(),
        "confidence_score": confidence,
        "warnings": [str(item) for item in warnings],
        "churn_acceleration": churn_acceleration,
        "risk_hint": str(signals.get("risk_hint") or "low").strip().lower(),
    }


def _growth_signal_snapshot(state: AgentState) -> dict[str, Any]:
    growth = _segmentation_or_node_payload(
        state,
        segmentation_key="growth_context",
        node_key="growth_data",
    )
    if not growth:
        return {
            "available": False,
            "status": "missing",
            "confidence_score": 1.0,
            "warnings": [],
            "short_growth": None,
            "mid_growth": None,
            "long_growth": None,
            "trend_acceleration": None,
            "insufficient_history": {},
        }

    horizons = growth.get("primary_horizons")
    if not isinstance(horizons, dict):
        horizons = {}
    insufficient = horizons.get("insufficient_history")
    if not isinstance(insufficient, dict):
        insufficient = {}

    def _num(value: Any) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    short_growth = _num(horizons.get("short_growth"))
    mid_growth = _num(horizons.get("mid_growth"))
    long_growth = _num(horizons.get("long_growth"))
    trend_acceleration = _num(horizons.get("trend_acceleration"))

    confidence_raw = growth.get("confidence_score")
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 1.0
    confidence = max(0.0, min(1.0, confidence))

    warnings = growth.get("warnings")
    if not isinstance(warnings, list):
        warnings = []

    available = any(value is not None for value in (short_growth, mid_growth, long_growth))
    return {
        "available": available,
        "status": str(growth.get("status") or "success").strip().lower(),
        "confidence_score": confidence,
        "warnings": [str(item) for item in warnings],
        "short_growth": short_growth,
        "mid_growth": mid_growth,
        "long_growth": long_growth,
        "trend_acceleration": trend_acceleration,
        "insufficient_history": {str(k): bool(v) for k, v in insufficient.items()},
    }


def _scenario_signal_snapshot(state: AgentState) -> dict[str, Any]:
    segmentation = payload_of(state.get("segmentation")) or {}
    scenario = segmentation.get("scenario_simulation") if isinstance(segmentation, dict) else None
    if not isinstance(scenario, dict):
        bundled = _segmentation_or_node_payload(
            state,
            segmentation_key="multivariate_scenario",
            node_key="multivariate_scenario_data",
        )
        if isinstance(bundled, dict):
            candidate = bundled.get("scenario_simulation")
            if isinstance(candidate, dict):
                scenario = candidate
    if not isinstance(scenario, dict):
        return {
            "available": False,
            "status": "missing",
            "confidence_score": 1.0,
            "warnings": [],
            "worst_growth": None,
            "best_growth": None,
            "worst_confidence_impact": None,
            "assumptions": {},
            "insufficient_history": {},
        }

    scenarios = scenario.get("scenarios")
    if not isinstance(scenarios, Mapping):
        scenarios = {}
    worst = scenarios.get("worst")
    best = scenarios.get("best")
    if not isinstance(worst, Mapping):
        worst = {}
    if not isinstance(best, Mapping):
        best = {}

    def _num(value: Any) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    worst_growth = _num(worst.get("projected_growth"))
    best_growth = _num(best.get("projected_growth"))

    assumptions = scenario.get("metadata")
    if isinstance(assumptions, Mapping):
        assumptions = assumptions.get("assumptions")
    if not isinstance(assumptions, Mapping):
        assumptions = {}

    worst_assumptions = worst.get("assumptions")
    if not isinstance(worst_assumptions, Mapping):
        worst_assumptions = {}
    worst_confidence_impact = _num(worst_assumptions.get("confidence_impact"))

    warnings = scenario.get("warnings")
    if not isinstance(warnings, list):
        warnings = []

    insufficient = scenario.get("insufficient_history")
    if not isinstance(insufficient, Mapping):
        insufficient = {}

    confidence_raw = scenario.get("base_confidence")
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 1.0
    confidence = max(0.0, min(1.0, confidence))

    return {
        "available": (worst_growth is not None) or (best_growth is not None),
        "status": str(scenario.get("status") or "success").strip().lower(),
        "confidence_score": confidence,
        "warnings": [str(item) for item in warnings],
        "worst_growth": worst_growth,
        "best_growth": best_growth,
        "worst_confidence_impact": worst_confidence_impact,
        "assumptions": dict(assumptions),
        "insufficient_history": {str(k): bool(v) for k, v in insufficient.items()},
    }


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
    kpi_key = _KPI_KEY_BY_BUSINESS_TYPE.get(business_type)

    if business_type not in _KPI_KEY_BY_BUSINESS_TYPE:
        risk_data = skipped(
            "unsupported_business_type",
            {"business_type": business_type},
        )
        return {**state, "risk_data": risk_data}

    kpi_envelope = state.get(kpi_key)
    kpi_status = status_of(kpi_envelope)
    kpi_payload: dict = _kpi_data_for_business_type(state)

    if kpi_status != "success":
        risk_data = skipped(
            "kpi_unavailable",
            {"business_type": business_type, "kpi_status": kpi_status},
        )
        return {**state, "risk_data": risk_data}

    if not kpi_payload:
        risk_data = skipped(
            "kpi_unavailable",
            {"business_type": business_type},
        )
        return {**state, "risk_data": risk_data}

    # Collect upstream diagnostics from KPI envelope.
    upstream_warnings: list[str] = warnings_of(kpi_envelope)
    upstream_confidence: float = confidence_of(kpi_envelope)

    try:
        kpi_signals = normalize_kpi_signals(kpi_payload)
    except Exception as exc:  # noqa: BLE001
        # Degrade to zero-signal baseline instead of hard failure.
        # The risk score will reflect only forecast/growth/cohort context,
        # and confidence will be heavily penalized.
        kpi_signals = {
            "revenue_growth_delta": 0.0,
            "churn_delta": 0.0,
            "conversion_delta": 0.0,
            "_warnings": [f"full kpi normalization failed, using zero baseline: {exc}"],
        }
        upstream_warnings.append(f"kpi_signal_normalization_degraded: {exc}")
        upstream_confidence = min(upstream_confidence, 0.3)

    # Propagate per-signal derivation warnings from the normalizer.
    normalizer_warnings = kpi_signals.pop("_warnings", [])
    if isinstance(normalizer_warnings, list) and normalizer_warnings:
        upstream_warnings.extend(normalizer_warnings)
        upstream_confidence = min(upstream_confidence, 0.5)

    forecast_state = state.get("forecast_data")
    forecast_payload: dict = payload_of(forecast_state) or {}
    forecast_status = status_of(forecast_state)
    cohort_snapshot = _cohort_signal_snapshot(state)
    growth_snapshot = _growth_signal_snapshot(state)
    scenario_snapshot = _scenario_signal_snapshot(state)

    if growth_snapshot["short_growth"] is not None:
        kpi_signals["revenue_growth_delta"] = float(growth_snapshot["short_growth"])

    forecast_signals: dict[str, float] = {}
    cohort_signals_used = False
    growth_signals_used = False
    scenario_signals_used = False
    forecast_context: dict[str, Any]
    if forecast_status == "success" and forecast_payload:
        try:
            forecast_signals = normalize_forecast_signals(forecast_payload)
            slope = float(forecast_signals.get("slope", 0.0))
            deviation = float(forecast_signals.get("deviation_percentage", 0.0))
            churn_accel = float(forecast_signals.get("churn_acceleration", 0.0))

            short_growth = growth_snapshot.get("short_growth")
            mid_growth = growth_snapshot.get("mid_growth")
            long_growth = growth_snapshot.get("long_growth")
            trend_accel = growth_snapshot.get("trend_acceleration")
            if mid_growth is not None:
                slope = (slope + float(mid_growth)) / 2.0
                growth_signals_used = True
            if short_growth is not None and long_growth is not None:
                deviation = max(deviation, abs(float(short_growth) - float(long_growth)))
                growth_signals_used = True
            if trend_accel is not None:
                growth_deacceleration_risk = max(0.0, -float(trend_accel))
                churn_accel = max(churn_accel, growth_deacceleration_risk)
                growth_signals_used = True
            worst_growth = scenario_snapshot.get("worst_growth")
            best_growth = scenario_snapshot.get("best_growth")
            if worst_growth is not None and best_growth is not None:
                deviation = max(deviation, abs(float(best_growth) - float(worst_growth)))
                scenario_signals_used = True

            forecast_context = {
                "status": "ok",
                "forecast_available": True,
                "slope": slope,
                "deviation_percentage": deviation,
                "churn_acceleration": churn_accel,
            }
        except Exception:
            forecast_context = {
                "status": "insufficient_data",
                "forecast_available": False,
            }
    else:
        short_growth = growth_snapshot.get("short_growth")
        mid_growth = growth_snapshot.get("mid_growth")
        long_growth = growth_snapshot.get("long_growth")
        trend_accel = growth_snapshot.get("trend_acceleration")
        growth_deacceleration_risk = (
            max(0.0, -float(trend_accel))
            if trend_accel is not None
            else 0.0
        )
        worst_growth = scenario_snapshot.get("worst_growth")
        best_growth = scenario_snapshot.get("best_growth")
        scenario_spread = (
            abs(float(best_growth) - float(worst_growth))
            if best_growth is not None and worst_growth is not None
            else 0.0
        )

        if cohort_snapshot["available"] or growth_snapshot["available"] or scenario_snapshot["available"]:
            cohort_signals_used = bool(cohort_snapshot["available"])
            growth_signals_used = bool(growth_snapshot["available"])
            scenario_signals_used = bool(scenario_snapshot["available"])
            cohort_accel = float(cohort_snapshot["churn_acceleration"] or 0.0)
            churn_accel = max(cohort_accel, growth_deacceleration_risk)
            slope = float(mid_growth or worst_growth or 0.0)
            deviation = (
                abs(float(short_growth) - float(long_growth))
                if short_growth is not None and long_growth is not None
                else 0.0
            )
            deviation = max(deviation, scenario_spread)
            forecast_context = {
                "status": (
                    "scenario_growth_cohort_proxy"
                    if scenario_snapshot["available"]
                    else ("growth_cohort_proxy" if growth_snapshot["available"] else "cohort_proxy")
                ),
                "forecast_available": True,
                "slope": slope,
                "deviation_percentage": deviation,
                "churn_acceleration": churn_accel,
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

        risk_payload: dict[str, Any] = {
            **result,
            "forecast_available": bool(forecast_context.get("forecast_available")),
            "cohort_signals_used": cohort_signals_used,
            "cohort_signal_status": cohort_snapshot.get("status"),
            "cohort_signal_confidence": cohort_snapshot.get("confidence_score"),
            "cohort_risk_hint": cohort_snapshot.get("risk_hint"),
            "growth_signals_used": growth_signals_used,
            "growth_signal_status": growth_snapshot.get("status"),
            "growth_signal_confidence": growth_snapshot.get("confidence_score"),
            "growth_short": growth_snapshot.get("short_growth"),
            "growth_mid": growth_snapshot.get("mid_growth"),
            "growth_long": growth_snapshot.get("long_growth"),
            "growth_trend_acceleration": growth_snapshot.get("trend_acceleration"),
            "growth_insufficient_history": growth_snapshot.get("insufficient_history"),
            "scenario_signals_used": scenario_signals_used,
            "scenario_signal_status": scenario_snapshot.get("status"),
            "scenario_signal_confidence": scenario_snapshot.get("confidence_score"),
            "scenario_worst_growth": scenario_snapshot.get("worst_growth"),
            "scenario_best_growth": scenario_snapshot.get("best_growth"),
            "scenario_worst_confidence_impact": scenario_snapshot.get("worst_confidence_impact"),
            "scenario_assumptions": scenario_snapshot.get("assumptions"),
            "scenario_insufficient_history": scenario_snapshot.get("insufficient_history"),
        }

        risk_warnings: list[str] = []

        if upstream_warnings or (0.0 < upstream_confidence < 1.0):
            risk_warnings.extend(
                [
                    f"Risk scored with degraded KPI confidence (confidence={upstream_confidence:.2f})",
                    *upstream_warnings,
                ]
            )

        if cohort_signals_used and cohort_snapshot.get("status") == "partial":
            cohort_conf = float(cohort_snapshot.get("confidence_score") or 0.5)
            risk_warnings.extend(
                [
                    f"Risk scored with partial cohort signals (confidence={cohort_conf:.2f})",
                    *cohort_snapshot.get("warnings", []),
                ]
            )

        if growth_signals_used and growth_snapshot.get("status") == "partial":
            growth_conf = float(growth_snapshot.get("confidence_score") or 0.5)
            risk_warnings.extend(
                [
                    f"Risk scored with partial growth signals (confidence={growth_conf:.2f})",
                    *growth_snapshot.get("warnings", []),
                ]
            )

        if scenario_signals_used and scenario_snapshot.get("status") == "partial":
            scenario_conf = float(scenario_snapshot.get("confidence_score") or 0.5)
            risk_warnings.extend(
                [
                    f"Risk scored with partial scenario signals (confidence={scenario_conf:.2f})",
                    *scenario_snapshot.get("warnings", []),
                ]
            )
        integrity = UnifiedSignalIntegrity.compute(state)
        integrity_score = max(0.0, min(1.0, float(integrity.get("overall_score") or 0.0)))
        risk_payload["signal_integrity"] = integrity
        risk_data = success(
            risk_payload,
            warnings=risk_warnings,
            confidence_score=integrity_score,
        )

    except Exception as exc:  # noqa: BLE001
        risk_data = failed(str(exc), {"entity_name": entity_name})

    return {**state, "risk_data": risk_data}
