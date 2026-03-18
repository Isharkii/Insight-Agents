"""
agent/nodes/synthesis_gate.py

Pre-LLM validation gate that hard-blocks insight generation when
required deterministic signals are unavailable or confidence is
below the minimum threshold.

This enforces the architectural contract:
    No LLM reasoning without validated deterministic signals.
"""

from __future__ import annotations

from typing import Any

from agent.graph_config import graph_node_config_for_business_type, signal_name_for_state_key
from agent.nodes.node_result import payload_of, status_of
from agent.signal_integrity import UnifiedSignalIntegrity
from agent.state import AgentState
from app.services.statistics.signal_conflict import apply_conflict_penalty
from llm_synthesis.schema import InsightOutput, set_self_analysis_mode

# Minimum deterministic confidence to allow LLM synthesis.
# Below this threshold the pipeline returns a structured failure.
MIN_CONFIDENCE_FOR_SYNTHESIS = 0.4


def _collect_required_failures(state: AgentState) -> list[str]:
    """Return names of required signals that hard-failed.

    Signals with ``insufficient_data`` or ``partial`` status are NOT failures —
    they produced best-effort output and should be gated by confidence (Gate 3),
    not by status (Gate 2).  Only truly absent or crashed signals block here.
    """
    config = graph_node_config_for_business_type(
        str(state.get("business_type") or ""),
    )
    # Statuses that represent a hard failure (no usable output produced).
    _HARD_FAIL_STATUSES = {"failed", "skipped"}

    failures: list[str] = []
    for key in config.required:
        value = state.get(key)
        if value is None:
            failures.append(signal_name_for_state_key(key))
        elif status_of(value) in _HARD_FAIL_STATUSES:
            failures.append(signal_name_for_state_key(key))
    return failures


def _compute_pre_synthesis_confidence(state: AgentState) -> float:
    """Compute deterministic confidence from signal envelopes before LLM runs."""
    integrity = UnifiedSignalIntegrity.compute(state)
    confidence = max(0.0, min(1.0, float(integrity.get("overall_score") or 0.0)))

    conflict_payload = payload_of(state.get("signal_conflicts")) or {}
    conflict_result = conflict_payload.get("conflict_result")
    if isinstance(conflict_result, dict):
        adjusted = apply_conflict_penalty(confidence, conflict_result, floor=0.0)
        confidence = max(
            0.0,
            min(1.0, float(adjusted.get("adjusted_confidence") or confidence)),
        )
    return confidence


def should_block_synthesis(state: AgentState) -> bool:
    """Return True if the pipeline must NOT proceed to LLM synthesis."""
    # Gate 1: pipeline_status explicitly failed
    pipeline_status = str(state.get("pipeline_status") or "").strip().lower()
    if pipeline_status == "failed":
        return True

    # Gate 2: any required signal missing or failed
    if _collect_required_failures(state):
        return True

    # Gate 3: deterministic confidence below threshold
    if _compute_pre_synthesis_confidence(state) < MIN_CONFIDENCE_FOR_SYNTHESIS:
        return True

    return False


def _has_competitor_data(state: AgentState) -> bool:
    """Read the deterministic competitive context contract from state."""
    ctx = state.get("competitive_context")
    if not isinstance(ctx, dict):
        return False
    return bool(ctx.get("available", False))


def build_blocked_response(state: AgentState) -> str:
    """Build a structured failure response when synthesis is blocked.

    Returns the serialized JSON string for ``state["final_response"]``.
    Sets self-analysis mode so the fallback text matches the data context.
    """
    has_competitors = _has_competitor_data(state)
    set_self_analysis_mode(not has_competitors)

    try:
        failed_signals = _collect_required_failures(state)
        pipeline_status = str(state.get("pipeline_status") or "failed").strip().lower()
        pre_confidence = _compute_pre_synthesis_confidence(state)

        parts: list[str] = []
        if failed_signals:
            parts.append(
                f"required signal(s) unavailable: {', '.join(failed_signals)}"
            )
        if pre_confidence < MIN_CONFIDENCE_FOR_SYNTHESIS:
            parts.append(
                f"deterministic confidence too low ({pre_confidence:.2f} < "
                f"{MIN_CONFIDENCE_FOR_SYNTHESIS})"
            )
        reason = (
            f"Insight generation blocked: {'; '.join(parts)}. "
            f"Pipeline status: {pipeline_status}. "
            f"The system requires validated deterministic signals "
            f"above the confidence threshold before LLM synthesis can execute."
        )

        failure = InsightOutput.failure(
            reason=reason,
            pipeline_status=pipeline_status if pipeline_status in ("success", "partial", "failed") else "failed",
        )

        return failure.model_dump_json()
    finally:
        set_self_analysis_mode(False)


def synthesis_gate_node(state: AgentState) -> AgentState:
    """LangGraph node: blocks LLM synthesis when required signals failed.

    When blocked, writes a structured failure to ``final_response`` and
    sets ``synthesis_blocked = True`` so the conditional edge can route
    directly to END, skipping the LLM node entirely.
    """
    integrity = UnifiedSignalIntegrity.compute(state)
    integrity_scores = UnifiedSignalIntegrity.score_vector_from_integrity(integrity)

    if should_block_synthesis(state):
        conflict_payload = payload_of(state.get("signal_conflicts")) or {}
        conflict_result = conflict_payload.get("conflict_result")
        warnings = ["Synthesis blocked by deterministic pre-checks."]
        # Surface reasoning warnings from integrity computation
        warnings.extend(integrity.get("reasoning_warnings", []))
        return {
            "synthesis_blocked": True,
            "final_response": build_blocked_response(state),
            "signal_integrity": integrity,
            "envelope_diagnostics": {
                "warnings": warnings,
                "missing_signal": _collect_required_failures(state),
                "isolated_layers": integrity.get("isolated_layers", []),
                "degraded_layers": integrity.get("degraded_layers", []),
                "reasoning_warnings": integrity.get("reasoning_warnings", []),
                "confidence_adjustments": integrity.get("confidence_adjustments", []),
                "confidence_score": _compute_pre_synthesis_confidence(state),
                "signal_conflicts": (
                    conflict_result if isinstance(conflict_result, dict) else {}
                ),
                "signal_integrity_scores": integrity_scores,
                "signal_integrity": integrity,
            },
        }

    return {
        "synthesis_blocked": False,
        "signal_integrity": integrity,
    }
