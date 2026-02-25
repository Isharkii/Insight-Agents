from __future__ import annotations

import json

from agent.nodes.llm_node import llm_node
from agent.nodes.node_result import failed, skipped, success
from llm_synthesis.schema import InsightOutput


def _synthetic_llm_output() -> InsightOutput:
    return InsightOutput(
        insight="Synthetic insight",
        evidence="Synthetic evidence",
        impact="Synthetic impact",
        recommended_action="Synthetic action",
        priority="medium",
        confidence_score=0.9,
        pipeline_status="success",
    )


def test_llm_node_optional_signal_outage_yields_partial_success(monkeypatch) -> None:
    monkeypatch.setattr("agent.nodes.llm_node.load_env_files", lambda: None)
    monkeypatch.setattr("agent.nodes.llm_node._build_adapter", lambda: object())
    monkeypatch.setattr(
        "agent.nodes.llm_node.generate_with_retry",
        lambda *_args, **_kwargs: _synthetic_llm_output(),
    )

    state = {
        "business_type": "saas",
        "entity_name": "acme",
        "saas_kpi_data": success({"records": [{"computed_kpis": {"mrr": {"value": 100.0}}}]}),
        "risk_data": success({"risk_score": 42.0}),
        "forecast_data": skipped("forecast_unavailable"),
        "root_cause": skipped("root_cause_unavailable"),
        "segmentation": success({"cohort_analytics": {"status": "success"}}),
        "prioritization": {
            "priority_level": "low",
            "recommended_focus": "monitor",
        },
    }

    result = llm_node(state)
    payload = json.loads(result["final_response"])

    assert result["pipeline_status"] == "partial"
    assert payload["pipeline_status"] == "partial"
    assert payload["insight"] == "Synthetic insight"
    assert payload["diagnostics"] is not None
    assert set(payload["diagnostics"]["missing_signal"]) == {"forecast", "root_cause"}
    assert payload["diagnostics"]["confidence_adjustments"]


def test_llm_node_required_signal_failure_is_structured_and_machine_readable(monkeypatch) -> None:
    monkeypatch.setattr("agent.nodes.llm_node.load_env_files", lambda: None)
    monkeypatch.setattr("agent.nodes.llm_node._build_adapter", lambda: object())
    monkeypatch.setattr(
        "agent.nodes.llm_node.generate_with_retry",
        lambda *_args, **_kwargs: _synthetic_llm_output(),
    )

    state = {
        "business_type": "saas",
        "entity_name": "acme",
        "saas_kpi_data": failed("missing_required_signals"),
        "risk_data": skipped("kpi_unavailable"),
        "forecast_data": skipped("forecast_unavailable"),
        "root_cause": skipped("root_cause_unavailable"),
    }

    result = llm_node(state)
    payload = json.loads(result["final_response"])

    assert result["pipeline_status"] == "failed"
    assert payload["pipeline_status"] == "failed"
    assert "saas_kpi" in payload["diagnostics"]["missing_signal"]
    assert payload["diagnostics"]["confidence_adjustments"]
