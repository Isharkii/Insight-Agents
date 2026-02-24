"""
agent/graph.py

LangGraph workflow assembly for the Insight Agent.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from langgraph.graph import END, START, StateGraph

from agent.nodes.agency_kpi_node import agency_kpi_fetch_node
from agent.nodes.business_router import business_router_node, route_by_business_type
from agent.nodes.ecommerce_kpi_node import ecommerce_kpi_fetch_node
from agent.nodes.intent import intent_node
from agent.nodes.llm_node import llm_node
from agent.nodes.node_result import failed, payload_of, skipped, status_of, success
from agent.nodes.risk_node import risk_node
from agent.nodes.saas_kpi_node import saas_kpi_fetch_node
from agent.state import AgentState
from app.services.role_performance_scoring import score_role_performance

_KPI_KEY_BY_BUSINESS_TYPE: dict[str, str] = {
    "saas": "saas_kpi_data",
    "ecommerce": "ecommerce_kpi_data",
    "agency": "agency_kpi_data",
}


def _resolve_kpi_payload(state: AgentState) -> dict[str, Any]:
    business_type = str(state.get("business_type") or "").lower()
    preferred_key = _KPI_KEY_BY_BUSINESS_TYPE.get(business_type)
    if preferred_key:
        preferred = state.get(preferred_key)
        if status_of(preferred) == "success":
            payload = payload_of(preferred)
            if isinstance(payload, dict):
                return payload

    for key in ("saas_kpi_data", "ecommerce_kpi_data", "agency_kpi_data"):
        candidate = state.get(key)
        if status_of(candidate) == "success":
            payload = payload_of(candidate)
            if isinstance(payload, dict):
                return payload
    return {}


def _extract_numeric_metric(
    computed_kpis: Mapping[str, Any],
    candidates: tuple[str, ...],
) -> float | None:
    for key in candidates:
        raw = computed_kpis.get(key)
        value = raw.get("value") if isinstance(raw, dict) else raw
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            return numeric
    return None


def role_analytics_node(state: AgentState) -> AgentState:
    """
    Role analytics stage inserted between KPI and risk.

    Internal execution:
      KPI -> Role Aggregation -> Statistical Engine -> Role Scoring.
    """
    try:
        kpi_payload = _resolve_kpi_payload(state)
        records = kpi_payload.get("records") if isinstance(kpi_payload, dict) else None
        if not isinstance(records, list) or not records:
            role_analytics_data = skipped("kpi_unavailable", {"records": 0})
            return {**state, "segmentation": role_analytics_data}

        has_role_column = any(isinstance(row, dict) and "role" in row for row in records)
        if not has_role_column:
            role_analytics_data = skipped("role_column_missing")
            return {**state, "segmentation": role_analytics_data}

        aggregated: dict[str, dict[str, list[float]]] = {}
        rows_with_role = 0

        for row in records:
            if not isinstance(row, dict):
                continue
            role = str(row.get("role") or "").strip()
            if not role:
                continue
            rows_with_role += 1

            computed_kpis = row.get("computed_kpis")
            if not isinstance(computed_kpis, dict):
                computed_kpis = {}

            growth = _extract_numeric_metric(
                computed_kpis,
                ("growth_rate", "revenue_growth_delta", "mrr_growth_rate"),
            )
            efficiency = _extract_numeric_metric(
                computed_kpis,
                (
                    "efficiency_metric",
                    "utilization_rate",
                    "conversion_rate",
                    "revenue_per_employee",
                ),
            )
            contribution = _extract_numeric_metric(
                computed_kpis,
                ("contribution_weight", "contribution", "revenue_share", "weight"),
            )

            slot = aggregated.setdefault(
                role,
                {
                    "growth_rate": [],
                    "efficiency_metric": [],
                    "contribution_weight": [],
                },
            )
            if growth is not None:
                slot["growth_rate"].append(growth)
            if efficiency is not None:
                slot["efficiency_metric"].append(efficiency)
            slot["contribution_weight"].append(
                contribution if contribution is not None else 1.0
            )

        if rows_with_role == 0 or not aggregated:
            role_analytics_data = skipped("role_values_missing")
            return {**state, "segmentation": role_analytics_data}

        statistical_engine: dict[str, dict[str, float | int]] = {}
        scoring_input: dict[str, dict[str, Any]] = {}
        for role, metrics in aggregated.items():
            growth_arr = np.asarray(metrics["growth_rate"], dtype=float)
            eff_arr = np.asarray(metrics["efficiency_metric"], dtype=float)
            contrib_arr = np.asarray(metrics["contribution_weight"], dtype=float)

            stability_arr = growth_arr if growth_arr.size > 1 else eff_arr
            variance = float(np.var(stability_arr)) if stability_arr.size > 1 else 0.0

            statistical_engine[role] = {
                "growth_mean": float(np.mean(growth_arr)) if growth_arr.size else 0.0,
                "efficiency_mean": float(np.mean(eff_arr)) if eff_arr.size else 0.0,
                "contribution_mean": float(np.mean(contrib_arr)) if contrib_arr.size else 0.0,
                "stability_variance": variance,
                "n_samples": int(
                    max(growth_arr.size, eff_arr.size, contrib_arr.size, 1)
                ),
            }

            scoring_input[role] = {
                "growth_rate": growth_arr.tolist(),
                "efficiency_metric": eff_arr.tolist(),
                "contribution_weight": contrib_arr.tolist(),
                "stability_series": stability_arr.tolist(),
            }

        category = str(state.get("business_type") or "").strip().lower() or None
        role_scores = score_role_performance(scoring_input, category=category)

        role_analytics_data = success(
            {
                "role_aggregation": aggregated,
                "statistical_engine": statistical_engine,
                "role_scoring": role_scores,
                "roles_scored": len(role_scores),
            }
        )
        return {**state, "segmentation": role_analytics_data}

    except Exception as exc:  # noqa: BLE001
        role_analytics_data = failed(str(exc), {"stage": "role_analytics_node"})
        return {**state, "segmentation": role_analytics_data}


def build_graph():
    """Build and compile the Insight Agent LangGraph workflow."""
    graph = StateGraph(AgentState)

    graph.add_node("intent", intent_node)
    graph.add_node("business_router", business_router_node)
    graph.add_node("saas_kpi_fetch", saas_kpi_fetch_node)
    graph.add_node("ecommerce_kpi_fetch", ecommerce_kpi_fetch_node)
    graph.add_node("agency_kpi_fetch", agency_kpi_fetch_node)
    graph.add_node("role_analytics", role_analytics_node)
    graph.add_node("risk", risk_node)
    graph.add_node("llm", llm_node)

    graph.add_edge(START, "intent")
    graph.add_edge("intent", "business_router")
    graph.add_conditional_edges(
        "business_router",
        route_by_business_type,
        {
            "saas_kpi_fetch": "saas_kpi_fetch",
            "ecommerce_kpi_fetch": "ecommerce_kpi_fetch",
            "agency_kpi_fetch": "agency_kpi_fetch",
        },
    )
    graph.add_edge("saas_kpi_fetch", "role_analytics")
    graph.add_edge("ecommerce_kpi_fetch", "role_analytics")
    graph.add_edge("agency_kpi_fetch", "role_analytics")
    graph.add_edge("role_analytics", "risk")
    graph.add_edge("risk", "llm")
    graph.add_edge("llm", END)

    return graph.compile()


insight_graph = build_graph()
