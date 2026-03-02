from __future__ import annotations

import json

from agent.nodes.llm_node import llm_node
from agent.nodes.node_result import failed, skipped, success
from llm_synthesis.schema import InsightOutput


def _synthetic_llm_output() -> InsightOutput:
    return InsightOutput(
        competitive_analysis={
            "summary": "Synthetic competitor summary from benchmark metrics.",
            "market_position": "Synthetic challenger position versus peers.",
            "relative_performance": "Synthetic growth metric trails competitor benchmark.",
            "key_advantages": ["Synthetic ARPU advantage versus competitor median."],
            "key_vulnerabilities": ["Synthetic churn weakness versus competitor benchmark."],
            "confidence": 0.9,
        },
        strategic_recommendations={
            "immediate_actions": ["Address synthetic competitor churn gap immediately."],
            "mid_term_moves": ["Reduce synthetic growth gap versus competitor benchmark."],
            "defensive_strategies": ["Defend synthetic segments where competitor strength is rising."],
            "offensive_strategies": ["Exploit synthetic competitor weakness in ARPU benchmark."],
        },
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
    assert payload["competitive_analysis"]["summary"].startswith("Synthetic competitor summary")
    assert payload["competitive_analysis"]["confidence"] < 0.9
    diagnostics = result.get("envelope_diagnostics") or {}
    assert set(diagnostics.get("missing_signal", [])) == {"forecast", "root_cause"}
    assert diagnostics.get("confidence_adjustments")


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
    assert payload["competitive_analysis"]["confidence"] <= 0.4
    assert "saas_kpi" in result["envelope_diagnostics"]["missing_signal"]
    assert result["envelope_diagnostics"]["confidence_adjustments"]
