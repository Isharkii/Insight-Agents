"""
agent/node_registry.py

Declarative node registration for the Insight Agent graph.

Each entry maps a graph node name to its callable and declares the edges
that should follow it.  ``build_graph()`` reads this registry instead of
hard-coding ``add_node`` / ``add_edge`` calls.

Adding a new node:
  1. Create ``agent/nodes/my_node.py`` with ``def my_node(state) -> AgentState``.
  2. Add an entry here in ``NODE_SEQUENCE`` at the right position.
  3. Done — no changes to ``graph.py`` needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agent.helpers.circuit_breaker import CircuitBreakerConfig, wrap_langgraph_node
from agent.helpers.node_observability import wrap_node_with_structured_logging
from agent.node_contracts import contract_spec_for_node
from agent.state import AgentState

# Type alias for a LangGraph node function.
NodeFn = Callable[[AgentState], AgentState]

_OUTPUT_KEY_BY_NODE: dict[str, str] = {
    "kpi_fetch": "kpi_data",
    "saas_kpi_fetch": "saas_kpi_data",
    "ecommerce_kpi_fetch": "ecommerce_kpi_data",
    "agency_kpi_fetch": "agency_kpi_data",
    "growth_engine": "growth_data",
    "timeseries_factors": "timeseries_factors_data",
    "cohort_analytics": "cohort_data",
    "category_formulas": "category_formula_data",
    "unit_economics": "unit_economics_data",
    "multivariate_scenario": "multivariate_scenario_data",
    "role_analytics": "segmentation",
    "forecast_fetch": "forecast_data",
    "benchmark": "benchmark_data",
    "signal_conflict": "signal_conflicts",
    "risk": "risk_data",
    "prioritization": "prioritization",
    "pipeline_status": "pipeline_status",
    "signal_enrichment": "signal_enrichment",
    "synthesis_gate": "final_response",
    "llm": "final_response",
}


@dataclass(frozen=True)
class NodeSpec:
    """Specification for a single graph node."""

    name: str
    fn: NodeFn
    next: str | None = None  # static successor (None = defined elsewhere)


def _lazy_import_nodes() -> dict[str, NodeFn]:
    """Import all node functions lazily to avoid circular imports."""
    from agent.nodes.agency_kpi_node import agency_kpi_fetch_node
    from agent.nodes.benchmark_node import benchmark_node
    from agent.nodes.business_router import business_router_node
    from agent.nodes.category_formula_node import category_formula_node
    from agent.nodes.cohort_analytics_node import cohort_analytics_node
    from agent.nodes.ecommerce_kpi_node import ecommerce_kpi_fetch_node
    from agent.nodes.forecast_node import forecast_fetch_node
    from agent.nodes.growth_engine_node import growth_engine_node
    from agent.nodes.intent import intent_node
    from agent.nodes.kpi_node import kpi_fetch_node
    from agent.nodes.llm_node import llm_node
    from agent.nodes.multivariate_scenario_node import multivariate_scenario_node
    from agent.nodes.prioritization_node import prioritization_node
    from agent.nodes.risk_node import risk_node
    from agent.nodes.role_analytics_node import role_analytics_node
    from agent.nodes.saas_kpi_node import saas_kpi_fetch_node
    from agent.nodes.signal_conflict_node import signal_conflict_node
    from agent.nodes.signal_enrichment_node import signal_enrichment_node
    from agent.nodes.synthesis_gate import synthesis_gate_node
    from agent.nodes.unit_economics_node import unit_economics_node
    from agent.pipeline_status import pipeline_status_node
    from agent.nodes.timeseries_factors_node import timeseries_factors_node

    guarded_forecast_fetch = wrap_langgraph_node(
        forecast_fetch_node,
        node_name="forecast_fetch",
        output_key="forecast_data",
        config=CircuitBreakerConfig(
            failure_threshold=3,
            cooldown_seconds=60.0,
            half_open_max_calls=1,
            degraded_confidence=0.2,
        ),
        degrade_on_failure=False,
    )

    def _signal_aggregation_barrier(state: AgentState) -> AgentState:
        """No-op barrier node: ensures role_analytics and forecast_fetch
        both complete before signal_conflict runs (prevents LangGraph
        superstep re-execution of downstream pipeline)."""
        return {}

    raw_nodes = {
        "intent": intent_node,
        "business_router": business_router_node,
        "kpi_fetch": kpi_fetch_node,
        "saas_kpi_fetch": saas_kpi_fetch_node,
        "ecommerce_kpi_fetch": ecommerce_kpi_fetch_node,
        "agency_kpi_fetch": agency_kpi_fetch_node,
        "growth_engine": growth_engine_node,
        "benchmark": benchmark_node,
        "timeseries_factors": timeseries_factors_node,
        "cohort_analytics": cohort_analytics_node,
        "category_formulas": category_formula_node,
        "unit_economics": unit_economics_node,
        "multivariate_scenario": multivariate_scenario_node,
        "role_analytics": role_analytics_node,
        "forecast_fetch": guarded_forecast_fetch,
        "signal_aggregation": _signal_aggregation_barrier,
        "signal_conflict": signal_conflict_node,
        "risk": risk_node,
        "prioritization": prioritization_node,
        "pipeline_status": pipeline_status_node,
        "signal_enrichment": signal_enrichment_node,
        "synthesis_gate": synthesis_gate_node,
        "llm": llm_node,
    }

    wrapped_nodes: dict[str, NodeFn] = {}
    for name, fn in raw_nodes.items():
        output_key = _OUTPUT_KEY_BY_NODE.get(name)
        contract = contract_spec_for_node(node_name=name, output_key=output_key)
        wrapped_nodes[name] = wrap_node_with_structured_logging(
            fn,
            node_name=name,
            output_key=output_key,
            input_contract=contract.input_model,
            output_contract=contract.output_model,
        )
    return wrapped_nodes


# ── Graph topology ──────────────────────────────────────────────────────────
#
# Phase 1 — Signal Extraction:
#   START → intent → business_router → [KPI branches] → growth_engine
#
# Phase 2 — Signal Enrichment (parallelizable fan-out from growth_engine):
#   growth_engine → timeseries_factors ─────────────────┐
#   growth_engine → cohort_analytics ───────────────────┤
#   growth_engine → category_formulas → unit_economics ─┤
#   growth_engine → multivariate_scenario ──────────────┤
#                                                       ↓
#                                                  role_analytics
#
# Phase 2b — Forecast (parallel with enrichment, needs only entity_name):
#   growth_engine → forecast_fetch ─────────────────────┐
#                                                       ↓
# Phase 3 — Signal Aggregation (fan-in via barrier):
#   role_analytics ─→ signal_aggregation ←─ forecast_fetch
#                          ↓
#                    signal_conflict
#
# Phase 4 — Decision Layer:
#   signal_conflict → risk → prioritization → pipeline_status → synthesis_gate
#
# Phase 5 — Synthesis:
#   synthesis_gate → [conditional] → llm → END
#

# Edges that form the enrichment fan-out from growth_engine.
ENRICHMENT_FAN_OUT: list[tuple[str, str]] = [
    ("growth_engine", "timeseries_factors"),
    ("growth_engine", "cohort_analytics"),
    ("growth_engine", "category_formulas"),
    ("growth_engine", "multivariate_scenario"),
    ("growth_engine", "forecast_fetch"),
    ("growth_engine", "benchmark"),
]

# Linear chains within the enrichment phase.
ENRICHMENT_CHAINS: list[tuple[str, str]] = [
    ("category_formulas", "unit_economics"),
]

# All enrichment nodes that must complete before role_analytics.
ENRICHMENT_FAN_IN_TO_ROLE: list[str] = [
    "timeseries_factors",
    "cohort_analytics",
    "unit_economics",
    "multivariate_scenario",
    "benchmark",
]

# All nodes that must complete before signal_aggregation barrier.
AGGREGATION_FAN_IN: list[str] = [
    "role_analytics",
    "forecast_fetch",
]

# The decision pipeline after signal aggregation.
DECISION_PIPELINE: list[tuple[str, str]] = [
    ("signal_aggregation", "signal_conflict"),
    ("signal_conflict", "risk"),
    ("risk", "prioritization"),
    ("prioritization", "pipeline_status"),
    ("pipeline_status", "signal_enrichment"),
    ("signal_enrichment", "synthesis_gate"),
]

# KPI branch nodes that converge into the linear pipeline.
KPI_BRANCHES: list[str] = [
    "kpi_fetch",
    "saas_kpi_fetch",
    "ecommerce_kpi_fetch",
    "agency_kpi_fetch",
]

# The convergence target for all KPI branches.
KPI_CONVERGENCE_NODE: str = "growth_engine"

# Full edge list for the post-KPI pipeline (used by graph.py).
INSIGHT_PIPELINE_SEQUENCE: list[tuple[str, str]] = [
    *ENRICHMENT_FAN_OUT,
    *ENRICHMENT_CHAINS,
    *[(node, "role_analytics") for node in ENRICHMENT_FAN_IN_TO_ROLE],
    *[(node, "signal_aggregation") for node in AGGREGATION_FAN_IN],
    *DECISION_PIPELINE,
]
