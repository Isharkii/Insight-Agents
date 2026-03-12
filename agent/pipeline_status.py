"""
agent/pipeline_status.py

Pipeline status derivation from required/optional node classification.
Extracted from graph.py.
"""

from __future__ import annotations

from agent.graph_config import graph_node_config_for_business_type
from agent.nodes.node_result import status_of
from agent.state import AgentState


def derive_pipeline_status(state: AgentState) -> str:
    """Compute pipeline status from required/optional node classification.

    Rules
    -----
    * ``"success"`` -- every *required* node produced ``status="success"``.
    * ``"partial"`` -- all required nodes succeeded, but at least one *optional*
      node that is wired did not succeed.
    * ``"failed"``  -- at least one required node is ``"skipped"`` or ``"failed"``.

    Unwired optional nodes (value is ``None`` in state because no graph
    node ever wrote to them) are silently ignored.
    """
    config = graph_node_config_for_business_type(str(state.get("business_type") or ""))
    required_keys = config.required
    optional_keys = config.optional

    for key in required_keys:
        if status_of(state.get(key)) != "success":
            return "failed"

    for key in optional_keys:
        value = state.get(key)
        if value is None:
            continue
        if status_of(value) != "success":
            return "partial"

    return "success"


def pipeline_status_node(state: AgentState) -> AgentState:
    """Pre-LLM node that computes ``pipeline_status`` from node outcomes."""
    return {"pipeline_status": derive_pipeline_status(state)}
