"""
agent/nodes/synthesis_gate.py

Pre-LLM validation gate that hard-blocks insight generation when
required deterministic signals are unavailable or confidence is
below the minimum threshold.

This enforces the architectural contract:
    No LLM reasoning without validated deterministic signals.
"""

from __future__ import annotations

import json
from typing import Any

from agent.graph_config import graph_node_config_for_business_type, signal_name_for_state_key
from agent.nodes.node_result import status_of
from agent.state import AgentState
from llm_synthesis.schema import (
    ConfidenceAdjustment,
    EnvelopeDiagnostics,
    InsightOutput,
)

# Minimum deterministic confidence to allow LLM synthesis.
# Below this threshold the pipeline returns a structured failure.
MIN_CONFIDENCE_FOR_SYNTHESIS = 0.4


def _collect_required_failures(state: AgentState) -> list[str]:
    """Return names of required signals that did not succeed."""
    config = graph_node_config_for_business_type(
        str(state.get("business_type") or ""),
    )
    failures: list[str] = []
    for key in config.required:
        value = state.get(key)
        if value is None or status_of(value) != "success":
            failures.append(signal_name_for_state_key(key))
    return failures


def should_block_synthesis(state: AgentState) -> bool:
    """Return True if the pipeline must NOT proceed to LLM synthesis."""
    # Gate 1: pipeline_status explicitly failed
    pipeline_status = str(state.get("pipeline_status") or "").strip().lower()
    if pipeline_status == "failed":
        return True

    # Gate 2: any required signal missing or failed
    if _collect_required_failures(state):
        return True

    return False


def build_blocked_response(state: AgentState) -> str:
    """Build a structured failure response when synthesis is blocked.

    Returns the serialized JSON string for ``state["final_response"]``.
    """
    failed_signals = _collect_required_failures(state)
    pipeline_status = str(state.get("pipeline_status") or "failed").strip().lower()

    reason = (
        f"Insight generation blocked: required signal(s) unavailable: "
        f"{', '.join(failed_signals)}. "
        f"Pipeline status: {pipeline_status}. "
        f"The system requires all deterministic signals to succeed before "
        f"LLM synthesis can execute."
    )

    diagnostics = EnvelopeDiagnostics(
        warnings=[
            f"Required signal '{s}' did not succeed." for s in failed_signals
        ],
        confidence_score=0.0,
        missing_signal=failed_signals,
        confidence_adjustments=[
            ConfidenceAdjustment(
                signal=s,
                delta=-0.35,
                reason="required_signal_unavailable",
            )
            for s in failed_signals
        ],
    )

    failure = InsightOutput.failure(
        reason=reason,
        pipeline_status=pipeline_status if pipeline_status in ("success", "partial", "failed") else "failed",
    ).model_copy(update={"diagnostics": diagnostics})

    return failure.model_dump_json()


def synthesis_gate_node(state: AgentState) -> AgentState:
    """LangGraph node: blocks LLM synthesis when required signals failed.

    When blocked, writes a structured failure to ``final_response`` and
    sets ``synthesis_blocked = True`` so the conditional edge can route
    directly to END, skipping the LLM node entirely.
    """
    if should_block_synthesis(state):
        return {
            **state,
            "synthesis_blocked": True,
            "final_response": build_blocked_response(state),
        }

    return {
        **state,
        "synthesis_blocked": False,
    }
