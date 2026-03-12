"""
agent/graph.py

LangGraph workflow assembly for the Insight Agent.
"""

from __future__ import annotations

from typing import Any

from agent.langgraph_compat import END, START, StateGraph
from agent.node_registry import (
    INSIGHT_PIPELINE_SEQUENCE,
    KPI_BRANCHES,
    KPI_CONVERGENCE_NODE,
    _lazy_import_nodes,
)
from agent.pipeline_status import derive_pipeline_status  # re-export
from agent.routing import route_after_synthesis_gate, route_by_business_type
from agent.state import AgentState

# Backward compatibility re-exports used by existing tests and callers.
from agent.helpers.canonical_queries import (  # noqa: F401
    filter_peers_by_size_band,
    filter_peers_by_size_band as _filter_peers_by_size_band,
    infer_size_band_from_metadata_rows,
    infer_size_band_from_metadata_rows as _infer_size_band_from_metadata_rows,
)
from agent.nodes import role_analytics_node as _role_analytics_module

_ORIGINAL_FETCH_CANONICAL_DIMENSION_ROWS = (
    _role_analytics_module.fetch_canonical_dimension_rows
)
_ORIGINAL_FETCH_MACRO_CONTEXT_ROWS = _role_analytics_module.fetch_macro_context_rows


def _fetch_canonical_dimension_rows(**kwargs: Any):
    return _ORIGINAL_FETCH_CANONICAL_DIMENSION_ROWS(**kwargs)


def _fetch_macro_context_rows(**kwargs: Any):
    return _ORIGINAL_FETCH_MACRO_CONTEXT_ROWS(**kwargs)


def role_analytics_node(state: AgentState) -> AgentState:
    """
    Compatibility wrapper around role_analytics_node.

    Tests and legacy call sites monkeypatch ``agent.graph._fetch_*`` helpers.
    Sync those hooks into the role analytics module before execution.
    """
    _role_analytics_module.fetch_canonical_dimension_rows = _fetch_canonical_dimension_rows
    _role_analytics_module.fetch_macro_context_rows = _fetch_macro_context_rows
    return _role_analytics_module.role_analytics_node(state)


def build_graph():
    """Build and compile the Insight Agent graph."""
    nodes = _lazy_import_nodes()
    graph = StateGraph(AgentState)

    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "intent")
    graph.add_edge("intent", "business_router")
    graph.add_conditional_edges(
        "business_router",
        route_by_business_type,
        {branch: branch for branch in KPI_BRANCHES},
    )

    for branch in KPI_BRANCHES:
        graph.add_edge(branch, KPI_CONVERGENCE_NODE)

    for source, target in INSIGHT_PIPELINE_SEQUENCE:
        graph.add_edge(source, target)

    graph.add_conditional_edges(
        "synthesis_gate",
        route_after_synthesis_gate,
        {"competitor_intelligence": "competitor_intelligence", "end": END},
    )
    graph.add_edge("competitor_intelligence", "llm")
    graph.add_edge("llm", END)
    return graph.compile()


insight_graph = build_graph()
