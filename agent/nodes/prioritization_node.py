"""
agent/nodes/prioritization_node.py

Deterministic prioritization node that combines risk score and root-cause
severity to produce a single focus recommendation.
"""

from __future__ import annotations

from typing import Any

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


def prioritization_node(state: AgentState) -> AgentState:
    """
    LangGraph node: prioritize the next focus area from risk and root-cause data.

    Writes:
        state["prioritization"] = {
            "priority_level": str,
            "recommended_focus": str,
        }
    """
    risk_data: dict[str, Any] = state.get("risk_data") or {}
    root_cause: dict[str, Any] = state.get("root_cause") or {}

    risk_score = max(0.0, min(100.0, _safe_float(risk_data.get("risk_score"))))
    root_severity = _normalize_severity(root_cause.get("severity"))
    risk_severity = _normalize_severity(risk_data.get("risk_level"))
    score_severity = _severity_from_score(risk_score)

    severities = [root_severity, risk_severity, score_severity]
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
        else f"monitor overall business risk ({int(round(risk_score))})"
    )

    prioritization = {
        "priority_level": top_severity,
        "recommended_focus": recommended_focus,
    }

    return {**state, "prioritization": prioritization}
