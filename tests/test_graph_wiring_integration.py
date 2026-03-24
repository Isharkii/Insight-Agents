from __future__ import annotations

from agent.graph import insight_graph


def test_graph_contains_deterministic_timeseries_pipeline_nodes() -> None:
    graph = insight_graph.get_graph()
    node_names = set(graph.nodes.keys())
    assert "growth_engine" in node_names
    assert "timeseries_factors" in node_names
    assert "cohort_analytics" in node_names
    assert "category_formulas" in node_names
    assert "unit_economics" in node_names
    assert "multivariate_scenario" in node_names
    assert "signal_conflict" in node_names
    assert "signal_enrichment" in node_names


def test_graph_wiring_order_matches_required_sequence() -> None:
    graph = insight_graph.get_graph()
    edges = {(edge.source, edge.target) for edge in graph.edges}

    # Phase 2: Enrichment fan-out from growth_engine.
    assert ("growth_engine", "timeseries_factors") in edges
    assert ("growth_engine", "cohort_analytics") in edges
    assert ("growth_engine", "category_formulas") in edges
    assert ("growth_engine", "multivariate_scenario") in edges
    assert ("growth_engine", "forecast_fetch") in edges

    # Enrichment chain: category_formulas → unit_economics.
    assert ("category_formulas", "unit_economics") in edges

    # Fan-in to role_analytics.
    assert ("timeseries_factors", "role_analytics") in edges
    assert ("cohort_analytics", "role_analytics") in edges
    assert ("unit_economics", "role_analytics") in edges
    assert ("multivariate_scenario", "role_analytics") in edges

    # Fan-in to signal_aggregation barrier, then signal_conflict.
    assert ("role_analytics", "signal_aggregation") in edges
    assert ("forecast_fetch", "signal_aggregation") in edges
    assert ("signal_aggregation", "signal_conflict") in edges

    # Decision pipeline.
    assert ("signal_conflict", "risk") in edges
    assert ("risk", "prioritization") in edges
    assert ("pipeline_status", "signal_enrichment") in edges
    assert ("signal_enrichment", "synthesis_gate") in edges

    # Synthesis: synthesis_gate routes conditionally to llm or END.
    assert ("synthesis_gate", "llm") in edges
