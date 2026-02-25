"""
agent/nodes/prioritization_node.py

Deterministic prioritization node that combines risk score and root-cause
severity to produce a single focus recommendation.
"""

from __future__ import annotations

from typing import Any

from agent.nodes.node_result import payload_of
from agent.state import AgentState

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
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


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


def _segmentation_or_node_payload(
    state: AgentState,
    *,
    segmentation_key: str,
    node_key: str,
) -> dict[str, Any]:
    segmentation: dict[str, Any] = payload_of(state.get("segmentation")) or {}
    if isinstance(segmentation, dict):
        candidate = segmentation.get(segmentation_key)
        if isinstance(candidate, dict):
            return candidate
    candidate = payload_of(state.get(node_key))
    return candidate if isinstance(candidate, dict) else {}


def _cohort_signals(state: AgentState) -> dict[str, Any]:
    cohort = _segmentation_or_node_payload(
        state,
        segmentation_key="cohort_analytics",
        node_key="cohort_data",
    )
    if not cohort:
        return {
            "status": "missing",
            "confidence_score": 1.0,
            "risk_hint": "low",
            "retention_decay": 0.0,
            "churn_acceleration": 0.0,
            "worst_cohort": None,
        }

    signals = cohort.get("signals")
    if not isinstance(signals, dict):
        signals = {}

    try:
        confidence = float(cohort.get("confidence_score", 1.0))
    except (TypeError, ValueError):
        confidence = 1.0
    confidence = max(0.0, min(1.0, confidence))

    return {
        "status": str(cohort.get("status") or "partial").strip().lower(),
        "confidence_score": confidence,
        "risk_hint": str(signals.get("risk_hint") or "low").strip().lower(),
        "retention_decay": _safe_float(signals.get("retention_decay")),
        "churn_acceleration": _safe_float(signals.get("churn_acceleration")),
        "worst_cohort": signals.get("worst_cohort"),
    }


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
    growth = _segmentation_or_node_payload(
        state,
        segmentation_key="growth_context",
        node_key="growth_data",
    )
    if not growth:
        return {
            "status": "missing",
            "confidence_score": 1.0,
            "short_growth": 0.0,
            "mid_growth": 0.0,
            "long_growth": 0.0,
            "trend_acceleration": 0.0,
            "insufficient_history": {},
        }

    horizons = growth.get("primary_horizons")
    if not isinstance(horizons, dict):
        horizons = {}
    insufficient = horizons.get("insufficient_history")
    if not isinstance(insufficient, dict):
        insufficient = {}

    try:
        confidence = float(growth.get("confidence_score", 1.0))
    except (TypeError, ValueError):
        confidence = 1.0
    confidence = max(0.0, min(1.0, confidence))

    return {
        "status": str(growth.get("status") or "partial").strip().lower(),
        "confidence_score": confidence,
        "short_growth": _safe_float(horizons.get("short_growth")),
        "mid_growth": _safe_float(horizons.get("mid_growth")),
        "long_growth": _safe_float(horizons.get("long_growth")),
        "trend_acceleration": _safe_float(horizons.get("trend_acceleration")),
        "insufficient_history": {str(k): bool(v) for k, v in insufficient.items()},
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
    segmentation: dict[str, Any] = payload_of(state.get("segmentation")) or {}
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
            "status": "missing",
            "base_confidence": 1.0,
            "worst_growth": 0.0,
            "best_growth": 0.0,
            "worst_confidence_impact": 0.0,
            "assumptions": {},
        }

    scenarios = scenario.get("scenarios")
    if not isinstance(scenarios, dict):
        scenarios = {}
    worst = scenarios.get("worst")
    best = scenarios.get("best")
    if not isinstance(worst, dict):
        worst = {}
    if not isinstance(best, dict):
        best = {}

    metadata = scenario.get("metadata")
    if isinstance(metadata, dict):
        assumptions = metadata.get("assumptions")
    else:
        assumptions = {}
    if not isinstance(assumptions, dict):
        assumptions = {}

    return {
        "status": str(scenario.get("status") or "partial").strip().lower(),
        "base_confidence": max(0.0, min(1.0, _safe_float(scenario.get("base_confidence", 1.0)))),
        "worst_growth": _safe_float(worst.get("projected_growth")),
        "best_growth": _safe_float(best.get("projected_growth")),
        "worst_confidence_impact": _safe_float(_as_dict(worst.get("assumptions")).get("confidence_impact")),
        "assumptions": assumptions,
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


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def prioritization_node(state: AgentState) -> AgentState:
    """
    LangGraph node: prioritize the next focus area from risk and root-cause data.

    Writes:
        state["prioritization"] = {
            "priority_level": str,
            "recommended_focus": str,
        }
    """
    risk_data: dict[str, Any] = payload_of(state.get("risk_data")) or {}
    root_cause: dict[str, Any] = payload_of(state.get("root_cause")) or {}
    cohort_snapshot = _cohort_signals(state)
    growth_snapshot = _growth_signals(state)
    scenario_snapshot = _scenario_signals(state)

    risk_score = max(0.0, min(100.0, _safe_float(risk_data.get("risk_score"))))
    root_severity = _normalize_severity(root_cause.get("severity"))
    risk_severity = _normalize_severity(risk_data.get("risk_level"))
    score_severity = _severity_from_score(risk_score)
    cohort_severity = _severity_from_cohort(cohort_snapshot)
    growth_severity = _severity_from_growth(growth_snapshot)
    scenario_severity = _severity_from_scenario(scenario_snapshot)

    severities = [
        root_severity,
        risk_severity,
        score_severity,
        cohort_severity,
        growth_severity,
        scenario_severity,
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

    prioritization = {
        "priority_level": top_severity,
        "recommended_focus": recommended_focus,
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
    }

    return {**state, "prioritization": prioritization}
