"""
agent/nodes/prioritization_node.py

Deterministic prioritization node that combines risk score and root-cause
severity to produce a single focus recommendation.
"""

from __future__ import annotations

from typing import Any

from agent.helpers.confidence_model import (
    compute_standard_confidence,
    propagate_reasoning_strategy_confidence,
)
from agent.helpers.signal_snapshots import (
    benchmark_signal_snapshot,
    cohort_signal_snapshot,
    growth_signal_snapshot,
    scenario_signal_snapshot,
    signal_conflict_snapshot,
    _safe_float as _snapshot_safe_float,
)
from agent.nodes.node_result import confidence_of, payload_of
from agent.signal_integrity import UnifiedSignalIntegrity
from agent.state import AgentState
from app.services.statistics.signal_conflict import build_uncertainty_summary

_SEVERITY_RANK: dict[str, int] = {
    "critical": 4,
    "high": 3,
    "moderate": 2,
    "low": 1,
}

_SCORE_TO_LEVEL: list[tuple[int, str]] = [
    (80, "critical"),
    (60, "high"),
    (30, "moderate"),
    (0, "low"),
]


def _normalize_severity(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in _SEVERITY_RANK:
        return text
    if text == "medium":
        return "moderate"
    return "low"


def _severity_from_score(risk_score: float) -> str:
    for threshold, level in _SCORE_TO_LEVEL:
        if risk_score >= threshold:
            return level
    return "low"


def _safe_float(value: Any) -> float:
    return _snapshot_safe_float(value)


def _issue_candidates(root_cause: dict[str, Any]) -> list[str]:
    primary = str(root_cause.get("primary_issue") or "").strip()
    if primary and primary != "no_issue_detected":
        candidates = [primary]
    else:
        candidates = []

    for item in root_cause.get("contributing_factors") or []:
        issue = str(item or "").strip()
        if issue and issue not in candidates:
            candidates.append(issue)

    for item in root_cause.get("root_causes") or []:
        issue = str(item or "").strip()
        if issue and issue not in candidates:
            candidates.append(issue)

    return candidates


def _to_focus_text(issue: str) -> str:
    return issue.replace("_", " ")


def _cohort_signals(state: AgentState) -> dict[str, Any]:
    return cohort_signal_snapshot(state)


def _severity_from_cohort(snapshot: dict[str, Any]) -> str:
    hint = str(snapshot.get("risk_hint") or "").strip().lower()
    if hint in _SEVERITY_RANK:
        return hint

    decay = _safe_float(snapshot.get("retention_decay"))
    acceleration = _safe_float(snapshot.get("churn_acceleration"))
    if decay >= 0.10 or acceleration >= 0.05:
        return "high"
    if decay >= 0.05 or acceleration >= 0.02:
        return "moderate"
    return "low"


def _cohort_focus_text(snapshot: dict[str, Any]) -> str | None:
    worst = snapshot.get("worst_cohort")
    if not isinstance(worst, dict):
        return None
    key = str(worst.get("cohort_key") or "").strip()
    value = str(worst.get("cohort_value") or "").strip()
    if not key or not value:
        return None
    return f"stabilize retention for cohort {key}={value}"


def _growth_signals(state: AgentState) -> dict[str, Any]:
    snapshot = growth_signal_snapshot(state)
    # Prioritization uses 0.0 defaults for None values (severity comparisons).
    return {
        **snapshot,
        "short_growth": _safe_float(snapshot.get("short_growth")),
        "mid_growth": _safe_float(snapshot.get("mid_growth")),
        "long_growth": _safe_float(snapshot.get("long_growth")),
        "trend_acceleration": _safe_float(snapshot.get("trend_acceleration")),
    }


def _severity_from_growth(snapshot: dict[str, Any]) -> str:
    short_growth = _safe_float(snapshot.get("short_growth"))
    mid_growth = _safe_float(snapshot.get("mid_growth"))
    long_growth = _safe_float(snapshot.get("long_growth"))
    trend_acc = _safe_float(snapshot.get("trend_acceleration"))

    min_growth = min(short_growth, mid_growth, long_growth)
    if min_growth <= -0.10 or trend_acc <= -0.05:
        return "high"
    if min_growth <= -0.03 or trend_acc <= -0.02:
        return "moderate"
    return "low"


def _growth_focus_text(snapshot: dict[str, Any]) -> str | None:
    short_growth = _safe_float(snapshot.get("short_growth"))
    mid_growth = _safe_float(snapshot.get("mid_growth"))
    long_growth = _safe_float(snapshot.get("long_growth"))
    if short_growth < 0.0 and mid_growth < 0.0 and long_growth < 0.0:
        return "reverse multi-horizon growth decline"
    if short_growth < 0.0 and mid_growth >= 0.0:
        return "stabilize near-term growth softness"
    return None


def _scenario_signals(state: AgentState) -> dict[str, Any]:
    snapshot = scenario_signal_snapshot(state)
    # Prioritization uses 0.0 defaults for None values (severity comparisons).
    return {
        **snapshot,
        "worst_growth": _safe_float(snapshot.get("worst_growth")),
        "best_growth": _safe_float(snapshot.get("best_growth")),
        "worst_confidence_impact": _safe_float(snapshot.get("worst_confidence_impact")),
    }


def _severity_from_scenario(snapshot: dict[str, Any]) -> str:
    worst_growth = _safe_float(snapshot.get("worst_growth"))
    if worst_growth <= -0.10:
        return "high"
    if worst_growth <= -0.03:
        return "moderate"
    return "low"


def _scenario_focus_text(snapshot: dict[str, Any]) -> str | None:
    worst_growth = _safe_float(snapshot.get("worst_growth"))
    if worst_growth <= -0.10:
        return "mitigate worst-case downside scenario assumptions"
    if worst_growth <= -0.03:
        return "prepare contingency plan for downside scenario"
    return None



def _signal_conflict_snapshot(state: AgentState) -> dict[str, Any]:
    return signal_conflict_snapshot(state)


def prioritization_node(state: AgentState) -> AgentState:
    """
    LangGraph node: prioritize the next focus area from risk and root-cause data.

    Writes:
        state["prioritization"] = {
            "priority_level": str,
            "recommended_focus": str,
        }
    """
    risk_envelope = state.get("risk_data")
    risk_data: dict[str, Any] = payload_of(risk_envelope) or {}
    root_cause: dict[str, Any] = payload_of(state.get("root_cause")) or {}
    cohort_snapshot = _cohort_signals(state)
    growth_snapshot = _growth_signals(state)
    scenario_snapshot = _scenario_signals(state)
    conflict_snapshot = _signal_conflict_snapshot(state)
    benchmark_snapshot = benchmark_signal_snapshot(state)

    risk_score = max(0.0, min(100.0, _safe_float(risk_data.get("risk_score"))))
    root_severity = _normalize_severity(root_cause.get("severity"))
    risk_severity = _normalize_severity(risk_data.get("risk_level"))
    score_severity = _severity_from_score(risk_score)
    cohort_severity = _severity_from_cohort(cohort_snapshot)
    growth_severity = _severity_from_growth(growth_snapshot)
    scenario_severity = _severity_from_scenario(scenario_snapshot)

    # Derive conflict-based severity: unresolved signal conflicts
    # represent contradictory intelligence and must escalate priority.
    _conflict_count = int(conflict_snapshot.get("conflict_count") or 0)
    _conflict_uncertainty = bool(conflict_snapshot.get("uncertainty_flag", False))
    if _conflict_uncertainty or _conflict_count >= 3:
        conflict_severity = "critical"
    elif _conflict_count >= 2:
        conflict_severity = "high"
    elif _conflict_count >= 1:
        conflict_severity = "moderate"
    else:
        conflict_severity = "low"

    severities = [
        root_severity,
        risk_severity,
        score_severity,
        cohort_severity,
        growth_severity,
        scenario_severity,
        conflict_severity,
    ]
    top_severity = max(severities, key=lambda level: _SEVERITY_RANK[level])

    issues = _issue_candidates(root_cause)
    ranked = sorted(
        issues,
        key=lambda issue: (
            -_SEVERITY_RANK[top_severity],
            -risk_score,
            issue,
        ),
    )

    # ── Insight quality classification ─────────────────────────────
    # Determines whether recommendations should be produced.
    # partial_insight (< 3 valid layers) → strip recommendations
    try:
        _integrity = UnifiedSignalIntegrity.compute(state)
        _insight_quality = str(_integrity.get("insight_quality", "full_insight"))
        _layer_classification = _integrity.get("layer_classification", {})
    except Exception:
        _insight_quality = "full_insight"
        _layer_classification = {}

    # ── Uncertainty mode ────────────────────────────────────────────
    # Uncertainty mode mirrors synthesis hard-gating semantics:
    #   - high-severity conflict present + cumulative severity > 1.0, OR
    #   - 3+ conflicts regardless of individual severity.
    # This avoids withholding recommendations for moderate two-conflict
    # cases where signals are noisy but still actionable.
    _total_severity = _safe_float(conflict_snapshot.get("total_severity"))
    uncertainty_mode = (
        (_conflict_uncertainty and _total_severity > 1.0)
        or _conflict_count >= 3
    )

    if _insight_quality == "partial_insight":
        recommended_focus = (
            "partial insight — insufficient analytical layers for "
            "recommendations; additional data sources required"
        )
        # Downgrade priority to low — cannot justify high priority
        # from incomplete signal coverage
        top_severity = "low"
    elif uncertainty_mode:
        recommended_focus = (
            "conflicting signals detected — recommendations withheld "
            "pending resolution; require additional data or resolution"
        )
    else:
        recommended_focus = (
            _to_focus_text(ranked[0])
            if ranked
            else (
                _cohort_focus_text(cohort_snapshot)
                or _growth_focus_text(growth_snapshot)
                or _scenario_focus_text(scenario_snapshot)
                or f"monitor overall business risk ({int(round(risk_score))})"
            )
        )
        if conflict_snapshot["conflict_count"] > 0:
            recommended_focus = (
                f"{recommended_focus}; resolve conflicting signals before executing strategy"
            )

    # ── Benchmark-informed recommendations ────────────────────────
    # When benchmark data is available, identify weakest/strongest
    # metrics relative to peers and generate specific actions.
    benchmark_recommendations: list[str] = []
    if (
        benchmark_snapshot.get("status") in ("success", "partial")
        and _insight_quality != "partial_insight"
        and not uncertainty_mode
    ):
        weakest = benchmark_snapshot.get("weakest_metrics") or []
        strongest = benchmark_snapshot.get("strongest_metrics") or []
        position = benchmark_snapshot.get("market_position")

        for weak in weakest[:2]:
            metric = str(weak.get("metric", "")).replace("_", " ")
            pct = weak.get("percentile", 0)
            benchmark_recommendations.append(
                f"improve {metric} (peer percentile: {pct:.0f}%)"
            )

        for strong in strongest[:1]:
            metric = str(strong.get("metric", "")).replace("_", " ")
            pct = strong.get("percentile", 100)
            benchmark_recommendations.append(
                f"leverage {metric} advantage (peer percentile: {pct:.0f}%)"
            )

        if position == "Declining":
            benchmark_recommendations.append(
                "address declining market position relative to peers"
            )
        elif position == "Challenger":
            benchmark_recommendations.append(
                "sustain growth momentum to close gap with market leaders"
            )

    reasoning_confidence_input = _safe_float(
        risk_data.get("reasoning_confidence_score")
    )
    if reasoning_confidence_input <= 0.0:
        reasoning_confidence_input = confidence_of(risk_envelope)

    strategy_model = compute_standard_confidence(
        values=[
            risk_score,
            float(_SEVERITY_RANK.get(root_severity, 1)),
            float(_SEVERITY_RANK.get(risk_severity, 1)),
            float(_SEVERITY_RANK.get(cohort_severity, 1)),
            float(_SEVERITY_RANK.get(growth_severity, 1)),
            float(_SEVERITY_RANK.get(scenario_severity, 1)),
            float(conflict_snapshot.get("conflict_count") or 0.0),
        ],
        signals={
            "cohort_retention_decay": -_safe_float(cohort_snapshot.get("retention_decay")),
            "cohort_churn_acceleration": -_safe_float(cohort_snapshot.get("churn_acceleration")),
            "growth_short": growth_snapshot.get("short_growth"),
            "growth_mid": growth_snapshot.get("mid_growth"),
            "growth_long": growth_snapshot.get("long_growth"),
            "growth_trend_acceleration": growth_snapshot.get("trend_acceleration"),
            "scenario_worst_growth": scenario_snapshot.get("worst_growth"),
            "scenario_best_growth": scenario_snapshot.get("best_growth"),
            "signal_conflict_count": -_safe_float(conflict_snapshot.get("conflict_count")),
            "signal_conflict_severity": -_safe_float(conflict_snapshot.get("total_severity")),
        },
        dataset_confidence=1.0,
        upstream_confidences=[
            reasoning_confidence_input,
            _safe_float(cohort_snapshot.get("confidence_score")),
            _safe_float(growth_snapshot.get("confidence_score")),
            _safe_float(scenario_snapshot.get("base_confidence")),
        ],
        status="success",
    )
    uncertainty_penalty = min(0.20, _safe_float(conflict_snapshot.get("confidence_penalty")))
    propagation = propagate_reasoning_strategy_confidence(
        insight_confidence=float(strategy_model["confidence_score"]),
        reasoning_confidence=reasoning_confidence_input,
        strategy_penalty=(
            (0.10 if top_severity in {"high", "critical"} else 0.0)
            + uncertainty_penalty
        ),
    )

    # Build conflict summary for uncertainty mode
    _conflict_summary = None
    if uncertainty_mode:
        _raw_conflict = conflict_snapshot.get("_raw_conflict_result")
        if isinstance(_raw_conflict, dict):
            _conflict_summary = build_uncertainty_summary(_raw_conflict)
        else:
            _conflict_summary = {
                "conflict_summary": (
                    f"{_conflict_count} signal conflict(s) with severity "
                    f"{_total_severity:.2f}. Recommendations withheld."
                ),
                "affected_signals": [],
                "decision": "withheld",
            }

    prioritization = {
        "priority_level": top_severity,
        "recommended_focus": recommended_focus,
        "insight_quality": _insight_quality,
        "layer_classification": _layer_classification,
        "uncertainty_mode": uncertainty_mode,
        "decision": (
            "withheld" if uncertainty_mode or _insight_quality == "partial_insight"
            else "active"
        ),
        "conflict_summary": _conflict_summary,
        "reasoning_confidence_score": propagation["reasoning_confidence"],
        "confidence_score": propagation["strategy_confidence"],
        "confidence_breakdown": strategy_model,
        "confidence_propagation": propagation,
        "cohort_signal_used": cohort_snapshot.get("status") in {"success", "partial"},
        "cohort_signal_confidence": cohort_snapshot.get("confidence_score"),
        "cohort_risk_hint": cohort_snapshot.get("risk_hint"),
        "growth_signal_used": growth_snapshot.get("status") in {"success", "partial"},
        "growth_signal_confidence": growth_snapshot.get("confidence_score"),
        "growth_short": growth_snapshot.get("short_growth"),
        "growth_mid": growth_snapshot.get("mid_growth"),
        "growth_long": growth_snapshot.get("long_growth"),
        "growth_trend_acceleration": growth_snapshot.get("trend_acceleration"),
        "growth_insufficient_history": growth_snapshot.get("insufficient_history"),
        "scenario_signal_used": scenario_snapshot.get("status") in {"success", "partial"},
        "scenario_base_confidence": scenario_snapshot.get("base_confidence"),
        "scenario_worst_growth": scenario_snapshot.get("worst_growth"),
        "scenario_best_growth": scenario_snapshot.get("best_growth"),
        "scenario_worst_confidence_impact": scenario_snapshot.get("worst_confidence_impact"),
        "scenario_assumptions": scenario_snapshot.get("assumptions"),
        "signal_conflict_status": conflict_snapshot.get("status"),
        "signal_conflict_count": conflict_snapshot.get("conflict_count"),
        "signal_conflict_total_severity": conflict_snapshot.get("total_severity"),
        "signal_conflict_confidence_penalty": conflict_snapshot.get("confidence_penalty"),
        "strategy_uncertainty_flag": conflict_snapshot.get("uncertainty_flag"),
        "signal_conflict_warnings": conflict_snapshot.get("warnings"),
        "benchmark_used": benchmark_snapshot.get("status") in ("success", "partial"),
        "benchmark_market_position": benchmark_snapshot.get("market_position"),
        "benchmark_overall_score": benchmark_snapshot.get("overall_score"),
        "benchmark_weakest_metrics": benchmark_snapshot.get("weakest_metrics", []),
        "benchmark_strongest_metrics": benchmark_snapshot.get("strongest_metrics", []),
        "benchmark_recommendations": benchmark_recommendations,
    }

    return {"prioritization": prioritization}
