"""
agent/graph.py

LangGraph workflow assembly for the Insight Agent.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent.state import AgentState
from agent.nodes.business_router import business_router_node
from agent.nodes.intent import intent_node
from agent.nodes.saas_kpi_node import saas_kpi_fetch_node
from agent.nodes.ecommerce_kpi_node import ecommerce_kpi_fetch_node
from agent.nodes.agency_kpi_node import agency_kpi_fetch_node
from agent.nodes.forecast_node import forecast_fetch_node
from agent.nodes.risk_node import risk_node
from agent.nodes.root_cause_node import root_cause_node
from agent.nodes.segmentation_node import segmentation_node
from agent.nodes.prioritization_node import prioritization_node
from agent.nodes.llm_node import llm_node


def build_graph():
    """
    Build and compile the Insight Agent LangGraph workflow.
    """
    graph = StateGraph(AgentState)

    graph.add_node("intent", intent_node)
    graph.add_node("business_router", business_router_node)
    graph.add_node("saas_kpi_fetch", saas_kpi_fetch_node)
    graph.add_node("ecommerce_kpi_fetch", ecommerce_kpi_fetch_node)
    graph.add_node("agency_kpi_fetch", agency_kpi_fetch_node)
    graph.add_node("forecast_fetch", forecast_fetch_node)
    graph.add_node("risk", risk_node)
    graph.add_node("root_cause", root_cause_node)
    graph.add_node("segmentation", segmentation_node)
    graph.add_node("prioritization", prioritization_node)
    graph.add_node("llm", llm_node)

    graph.add_edge(START, "intent")
    graph.add_edge("intent", "business_router")
    graph.add_conditional_edges(
        "business_router",
        business_router_node,
        {
            "saas_kpi_fetch": "saas_kpi_fetch",
            "ecommerce_kpi_fetch": "ecommerce_kpi_fetch",
            "agency_kpi_fetch": "agency_kpi_fetch",
        },
    )
    graph.add_edge("saas_kpi_fetch", "forecast_fetch")
    graph.add_edge("ecommerce_kpi_fetch", "forecast_fetch")
    graph.add_edge("agency_kpi_fetch", "forecast_fetch")
    graph.add_edge("forecast_fetch", "risk")
    graph.add_edge("risk", "root_cause")
    graph.add_edge("root_cause", "segmentation")
    graph.add_edge("segmentation", "prioritization")
    graph.add_edge("prioritization", "llm")
    graph.add_edge("llm", END)

    return graph.compile()


insight_graph = build_graph()
