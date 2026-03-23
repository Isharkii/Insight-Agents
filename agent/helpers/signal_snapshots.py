"""
agent/helpers/signal_snapshots.py

Shared signal snapshot extraction from state envelopes.

Used by risk_node and prioritization_node to read cohort, growth,
scenario, and signal-conflict signals from upstream node outputs
in a consistent, defensive manner.
"""

from __future__ import annotations

from typing import Any, Mapping

from agent.nodes.node_result import payload_of


def _safe_float(value: Any) -> float:
    """Coerce to float, defaulting to 0.0 on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_float_or_none(value: Any) -> float | None:
    """Coerce to float, returning None on failure."""
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _safe_confidence(value: Any) -> float:
    """Coerce a confidence value to [0.0, 1.0], defaulting to 1.0."""
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, min(1.0, conf))


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _segmentation_or_node_payload(
    state: dict[str, Any],
    *,
    segmentation_key: str,
    node_key: str,
) -> dict[str, Any]:
    """Resolve a signal payload from segmentation bundle or standalone node."""
    segmentation = payload_of(state.get("segmentation")) or {}
    if isinstance(segmentation, dict):
        candidate = segmentation.get(segmentation_key)
        if isinstance(candidate, dict):
            return candidate
    candidate = payload_of(state.get(node_key))
    return candidate if isinstance(candidate, dict) else {}


# ---------------------------------------------------------------------------
# Cohort snapshot
# ---------------------------------------------------------------------------

def cohort_signal_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    """Extract cohort analytics signals from state."""
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
            "retention_decay": 0.0,
            "risk_hint": "low",
            "worst_cohort": None,
        }

    signals = cohort.get("signals")
    if not isinstance(signals, dict):
        signals = {}

    confidence = _safe_confidence(cohort.get("confidence_score"))

    churn_acceleration = _safe_float_or_none(signals.get("churn_acceleration"))

    warnings = cohort.get("warnings")
    if not isinstance(warnings, list):
        warnings = []

    return {
        "available": churn_acceleration is not None,
        "status": str(cohort.get("status") or "success").strip().lower(),
        "confidence_score": confidence,
        "warnings": [str(item) for item in warnings],
        "churn_acceleration": churn_acceleration,
        "retention_decay": _safe_float(signals.get("retention_decay")),
        "risk_hint": str(signals.get("risk_hint") or "low").strip().lower(),
        "worst_cohort": signals.get("worst_cohort"),
    }


# ---------------------------------------------------------------------------
# Growth snapshot
# ---------------------------------------------------------------------------

def growth_signal_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    """Extract growth engine signals from state."""
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

    short_growth = _safe_float_or_none(horizons.get("short_growth"))
    mid_growth = _safe_float_or_none(horizons.get("mid_growth"))
    long_growth = _safe_float_or_none(horizons.get("long_growth"))
    trend_acceleration = _safe_float_or_none(horizons.get("trend_acceleration"))

    confidence = _safe_confidence(growth.get("confidence_score"))

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


# ---------------------------------------------------------------------------
# Scenario snapshot
# ---------------------------------------------------------------------------

def scenario_signal_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    """Extract multivariate scenario simulation signals from state."""
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
            "base_confidence": 1.0,
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

    worst_growth = _safe_float_or_none(worst.get("projected_growth"))
    best_growth = _safe_float_or_none(best.get("projected_growth"))

    assumptions = scenario.get("metadata")
    if isinstance(assumptions, Mapping):
        assumptions = assumptions.get("assumptions")
    if not isinstance(assumptions, Mapping):
        assumptions = {}

    worst_assumptions = worst.get("assumptions")
    if not isinstance(worst_assumptions, Mapping):
        worst_assumptions = {}
    worst_confidence_impact = _safe_float_or_none(
        worst_assumptions.get("confidence_impact")
    )

    warnings = scenario.get("warnings")
    if not isinstance(warnings, list):
        warnings = []

    insufficient = scenario.get("insufficient_history")
    if not isinstance(insufficient, Mapping):
        insufficient = {}

    confidence = _safe_confidence(scenario.get("base_confidence"))

    return {
        "available": (worst_growth is not None) or (best_growth is not None),
        "status": str(scenario.get("status") or "success").strip().lower(),
        "confidence_score": confidence,
        "base_confidence": confidence,
        "warnings": [str(item) for item in warnings],
        "worst_growth": worst_growth,
        "best_growth": best_growth,
        "worst_confidence_impact": worst_confidence_impact,
        "assumptions": dict(assumptions),
        "insufficient_history": {str(k): bool(v) for k, v in insufficient.items()},
    }


# ---------------------------------------------------------------------------
# Signal conflict snapshot
# ---------------------------------------------------------------------------

def benchmark_signal_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    """Extract benchmark intelligence from state for prioritization.

    Returns structured snapshot with composite scores, market position,
    and metric-level rankings for generating benchmark-informed recommendations.
    """
    envelope = state.get("benchmark_data")
    payload = payload_of(envelope) if isinstance(envelope, dict) else None
    if not isinstance(payload, dict):
        return {
            "status": "missing",
            "market_position": None,
            "overall_score": None,
            "growth_score": None,
            "stability_score": None,
            "weakest_metrics": [],
            "strongest_metrics": [],
            "peer_count": 0,
        }

    composite = payload.get("composite") or {}
    market_pos = payload.get("market_position") or {}
    ranking = payload.get("ranking") or {}

    overall = _safe_float_or_none(composite.get("overall_score"))
    growth = _safe_float_or_none(composite.get("growth_score"))
    stability = _safe_float_or_none(composite.get("stability_score"))

    # Extract metric-level rankings to identify weakest/strongest areas
    metric_rankings = ranking.get("metric_rankings") or {}
    scored_metrics: list[tuple[str, float]] = []
    for metric_name, metric_info in metric_rankings.items():
        if isinstance(metric_info, dict):
            percentile = _safe_float_or_none(metric_info.get("percentile_rank"))
            if percentile is not None:
                scored_metrics.append((str(metric_name), float(percentile)))

    scored_metrics.sort(key=lambda x: x[1])
    weakest = [
        {"metric": name, "percentile": round(pct, 2)}
        for name, pct in scored_metrics[:3]
        if pct < 40.0
    ]
    strongest = [
        {"metric": name, "percentile": round(pct, 2)}
        for name, pct in reversed(scored_metrics)
        if pct >= 60.0
    ][:3]

    peer_selection = payload.get("peer_selection") or {}
    peer_count = len(peer_selection.get("selected_peers") or [])

    return {
        "status": "success" if overall is not None else "partial",
        "market_position": market_pos.get("position"),
        "position_confidence": _safe_float_or_none(market_pos.get("confidence")),
        "overall_score": overall,
        "growth_score": growth,
        "stability_score": stability,
        "weakest_metrics": weakest,
        "strongest_metrics": strongest,
        "peer_count": peer_count,
    }


def signal_conflict_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    """Extract signal conflict detection results from state."""
    envelope = state.get("signal_conflicts")
    payload = payload_of(envelope) if isinstance(envelope, dict) else None
    if not isinstance(payload, dict):
        return {
            "status": "missing",
            "conflict_count": 0,
            "total_severity": 0.0,
            "confidence_penalty": 0.0,
            "uncertainty_flag": False,
            "warnings": [],
        }

    conflict = payload.get("conflict_result")
    if not isinstance(conflict, dict):
        conflict = {}
    return {
        "status": str(conflict.get("status") or "missing").strip().lower(),
        "conflict_count": int(_safe_float(conflict.get("conflict_count"))),
        "total_severity": _safe_float(conflict.get("total_severity")),
        "confidence_penalty": _safe_float(conflict.get("confidence_penalty")),
        "uncertainty_flag": bool(conflict.get("uncertainty_flag", False)),
        "warnings": [
            str(item)
            for item in conflict.get("warnings", [])
            if str(item).strip()
        ]
        if isinstance(conflict.get("warnings"), list)
        else [],
        "_raw_conflict_result": conflict,
    }
