from __future__ import annotations

from agent.graph import insight_graph


def test_graph_contains_deterministic_timeseries_pipeline_nodes() -> None:
    graph = insight_graph.get_graph()
    node_names = set(graph.nodes.keys())
    assert "growth_engine" in node_names
    assert "timeseries_factors" in node_names
    assert "cohort_analytics" in node_names
    assert "category_formulas" in node_names
    assert "multivariate_scenario" in node_names
    assert "competitor_intelligence" in node_names


def test_graph_wiring_order_matches_required_sequence() -> None:
    graph = insight_graph.get_graph()
    edges = {(edge.source, edge.target) for edge in graph.edges}

    required_edges = {
        ("growth_engine", "timeseries_factors"),
        ("timeseries_factors", "cohort_analytics"),
        ("cohort_analytics", "category_formulas"),
        ("category_formulas", "multivariate_scenario"),
        ("multivariate_scenario", "role_analytics"),
        ("role_analytics", "forecast_fetch"),
        ("forecast_fetch", "risk"),
        ("risk", "prioritization"),
        ("pipeline_status", "synthesis_gate"),
        ("synthesis_gate", "competitor_intelligence"),
        ("competitor_intelligence", "llm"),
    }
    assert required_edges.issubset(edges)
