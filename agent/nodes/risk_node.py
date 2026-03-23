"""
agent/nodes/risk_node.py

Risk Node: normalizes signals from one business-specific KPI payload and
state.forecast_data, calls RiskOrchestrator, and stores the result in
state.risk_data.

No scoring math, no schema changes.
"""

from __future__ import annotations

from typing import Any, Mapping

from agent.helpers.confidence_model import compute_standard_confidence
from agent.helpers.signal_snapshots import (
    cohort_signal_snapshot,
    growth_signal_snapshot,
    scenario_signal_snapshot,
)
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
from app.services.statistics.signal_conflict import detect_conflicts, apply_conflict_penalty
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




# Snapshot extraction delegated to shared helpers.
_cohort_signal_snapshot = cohort_signal_snapshot
_growth_signal_snapshot = growth_signal_snapshot
_scenario_signal_snapshot = scenario_signal_snapshot


def _global_conflict_result(state: AgentState) -> tuple[dict[str, Any], str]:
    envelope = state.get("signal_conflicts")
    payload = payload_of(envelope) or {}
    result = payload.get("conflict_result")
    if isinstance(result, Mapping):
        return dict(result), "global"
    return {}, "local_fallback"


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
        return {"risk_data": risk_data}

    kpi_envelope = state.get(kpi_key)
    kpi_status = status_of(kpi_envelope)
    kpi_payload: dict = _kpi_data_for_business_type(state)

    if kpi_status != "success":
        risk_data = skipped(
            "kpi_unavailable",
            {"business_type": business_type, "kpi_status": kpi_status},
        )
        return {"risk_data": risk_data}

    if not kpi_payload:
        risk_data = skipped(
            "kpi_unavailable",
            {"business_type": business_type},
        )
        return {"risk_data": risk_data}

    # ── Isolation enforcement ─────────────────────────────────────
    # Compute integrity early so isolated layers can be excluded from
    # risk scoring.  Reused later for confidence scoring (line ~387).
    integrity = UnifiedSignalIntegrity.compute(state)
    _isolated_layers = set(integrity.get("isolated_layers", []))
    _forecast_isolated = "forecast" in _isolated_layers
    if _forecast_isolated:
        import logging as _risk_logging
        _risk_logging.getLogger("agent.nodes.risk").info(
            "Forecast layer isolated — excluding forecast signals from risk scoring"
        )

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

    # Track which signals were defaulted (phantom zeros) so we can
    # penalize confidence proportionally rather than treating them as real.
    defaulted_signals: list[str] = kpi_signals.pop("_defaulted", [])  # type: ignore[assignment]
    if not isinstance(defaulted_signals, list):
        defaulted_signals = []
    if defaulted_signals:
        # Apply confidence penalty: more defaulted signals = less reliable risk
        defaulted_penalty = min(0.3, 0.1 * len(defaulted_signals))
        upstream_confidence = max(0.1, upstream_confidence - defaulted_penalty)
        upstream_warnings.append(
            f"Risk scored with {len(defaulted_signals)} defaulted signal(s) "
            f"({', '.join(defaulted_signals)}); confidence penalized by {defaulted_penalty:.2f}."
        )

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
    if forecast_status == "success" and forecast_payload and not _forecast_isolated:
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

        # ── Signal coverage enforcement ─────────────────────────────
        # Count how many analytical dimensions contributed real data.
        # If fewer than 2 dimensions have real signal, the risk score
        # is computed on insufficient intelligence and confidence must
        # reflect that honestly.
        _dimension_status: dict[str, bool] = {
            "kpi": bool(kpi_signals.get("revenue_growth_delta") is not None
                        or kpi_signals.get("churn_delta") is not None),
            "forecast": bool(
                forecast_context.get("forecast_available")
                and forecast_context.get("status") not in (
                    "insufficient_data", "cohort_proxy",
                    "growth_cohort_proxy", "scenario_growth_cohort_proxy",
                )
            ),
            "cohort": cohort_signals_used,
            "growth": growth_signals_used,
            "scenario": scenario_signals_used,
        }
        _real_dimensions = sum(1 for v in _dimension_status.values() if v)
        _MIN_DIMENSIONS = 2
        _coverage_sufficient = _real_dimensions >= _MIN_DIMENSIONS

        if not _coverage_sufficient:
            # Severe penalty: risk computed from < 2 real dimensions.
            coverage_penalty = 0.3 + 0.1 * (_MIN_DIMENSIONS - _real_dimensions)
            upstream_confidence = max(0.05, upstream_confidence - coverage_penalty)
            upstream_warnings.append(
                f"Signal coverage insufficient: only {_real_dimensions}/{len(_dimension_status)} "
                f"analytical dimensions have real data "
                f"({', '.join(k for k, v in _dimension_status.items() if v) or 'none'}). "
                f"Confidence penalized by {coverage_penalty:.2f}."
            )

        risk_payload: dict[str, Any] = {
            **result,
            "signal_coverage": {
                "dimensions": _dimension_status,
                "real_count": _real_dimensions,
                "minimum_required": _MIN_DIMENSIONS,
                "sufficient": _coverage_sufficient,
            },
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
        # ── Signal conflict detection ────────────────────────────────────
        conflict_result, conflict_source = _global_conflict_result(state)
        if not conflict_result:
            conflict_result = detect_conflicts(kpi_signals)
        risk_payload["signal_conflicts"] = conflict_result
        risk_payload["signal_conflict_source"] = conflict_source
        if conflict_result.get("conflict_count", 0) > 0:
            risk_warnings.extend(conflict_result.get("warnings", []))

        # integrity already computed early for isolation enforcement
        integrity_score = max(0.0, min(1.0, float(integrity.get("overall_score") or 0.0)))

        # Apply conflict penalty to confidence
        if conflict_result.get("confidence_penalty", 0.0) > 0:
            adjustment = apply_conflict_penalty(integrity_score, conflict_result)
            integrity_score = adjustment["adjusted_confidence"]
            risk_payload["conflict_confidence_adjustment"] = adjustment

        confidence_values = [
            float(v)
            for v in kpi_signals.values()
            if isinstance(v, (int, float))
        ]
        if isinstance(forecast_context.get("slope"), (int, float)):
            confidence_values.append(float(forecast_context["slope"]))
        if isinstance(forecast_context.get("deviation_percentage"), (int, float)):
            confidence_values.append(float(forecast_context["deviation_percentage"]))
        if isinstance(forecast_context.get("churn_acceleration"), (int, float)):
            confidence_values.append(float(forecast_context["churn_acceleration"]))

        confidence_model = compute_standard_confidence(
            values=confidence_values,
            signals={
                **{f"kpi_{k}": v for k, v in kpi_signals.items()},
                "forecast_slope": forecast_context.get("slope"),
                "forecast_deviation": (
                    -abs(float(forecast_context.get("deviation_percentage")))
                    if isinstance(forecast_context.get("deviation_percentage"), (int, float))
                    else None
                ),
                "forecast_churn_acceleration": (
                    -float(forecast_context.get("churn_acceleration"))
                    if isinstance(forecast_context.get("churn_acceleration"), (int, float))
                    else None
                ),
                "signal_conflict_count": -float(conflict_result.get("conflict_count", 0)),
            },
            dataset_confidence=upstream_confidence,
            upstream_confidences=[
                float(cohort_snapshot.get("confidence_score") or 1.0),
                float(growth_snapshot.get("confidence_score") or 1.0),
                float(scenario_snapshot.get("confidence_score") or 1.0),
            ],
            status="success",
            base_warnings=risk_warnings,
        )
        reasoning_confidence = min(
            float(confidence_model["confidence_score"]),
            float(integrity_score),
        )

        risk_payload["signal_integrity"] = integrity
        risk_payload["confidence_breakdown"] = confidence_model
        risk_payload["reasoning_confidence_score"] = round(reasoning_confidence, 6)
        risk_data = success(
            risk_payload,
            warnings=confidence_model["warnings"],
            confidence_score=integrity_score,
        )

    except Exception as exc:  # noqa: BLE001
        risk_data = failed(str(exc), {"entity_name": entity_name})

    return {"risk_data": risk_data}
