from __future__ import annotations

import json

import pytest

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


def _synthetic_llm_output_max_confidence() -> InsightOutput:
    payload = _synthetic_llm_output().model_dump()
    payload["competitive_analysis"]["confidence"] = 1.0
    return InsightOutput.model_validate(payload)


def test_llm_node_optional_signal_outage_yields_partial_success(monkeypatch) -> None:
    monkeypatch.setattr("agent.nodes.llm_node.load_env_files", lambda: None)
    monkeypatch.setattr("agent.nodes.llm_node._build_adapter", lambda **_kw: object())
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
    assert "Synthetic competitor summary" in payload["competitive_analysis"]["summary"]
    assert payload["competitive_analysis"]["confidence"] < 0.9
    diagnostics = result.get("envelope_diagnostics") or {}
    assert set(diagnostics.get("missing_signal", [])) == {"forecast", "root_cause"}
    assert diagnostics.get("confidence_adjustments")
    integrity_scores = diagnostics.get("signal_integrity_scores") or {}
    assert set(integrity_scores.keys()) == {
        "KPI_score",
        "Forecast_score",
        "Competitive_score",
        "Cohort_score",
        "Segmentation_score",
        "Unified_integrity_score",
    }


def test_llm_node_required_signal_failure_is_structured_and_machine_readable(monkeypatch) -> None:
    monkeypatch.setattr("agent.nodes.llm_node.load_env_files", lambda: None)
    monkeypatch.setattr("agent.nodes.llm_node._build_adapter", lambda **_kw: object())
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


def test_llm_node_uses_unified_signal_integrity_for_confidence(monkeypatch) -> None:
    monkeypatch.setattr("agent.nodes.llm_node.load_env_files", lambda: None)
    monkeypatch.setattr("agent.nodes.llm_node._build_adapter", lambda **_kw: object())
    monkeypatch.setattr(
        "agent.nodes.llm_node.generate_with_retry",
        lambda *_args, **_kwargs: _synthetic_llm_output_max_confidence(),
    )

    state = {
        "business_type": "saas",
        "entity_name": "acme",
        "saas_kpi_data": success({"records": [{"computed_kpis": {"mrr": {"value": 100.0}}}]}),
        "risk_data": success({"risk_score": 42.0}),
        "competitive_context": {
            "available": True,
            "source": "external_fetch",
            "numeric_signals": [
                {
                    "metric_name": "growth_rate_mentioned_pct",
                    "unit": "ratio",
                    "sample_size": 4,
                    "mean": 0.2,
                }
            ],
        },
    }

    result = llm_node(state)
    payload = json.loads(result["final_response"])
    diagnostics = result.get("envelope_diagnostics") or {}
    adjustments = diagnostics.get("confidence_adjustments") or []
    reasons = {str(item.get("reason")) for item in adjustments if isinstance(item, dict)}

    assert "layer_integrity_contribution" in reasons
    # Only layers with score > 0 contribute to the mean.
    # KPI layer score ~0.333333 is the sole scoring layer here
    # (competitive has score=0 due to 0 peers, excluded from mean).
    # With only 1 valid layer (< 3 required), the confidence cap
    # reduces the base from 0.333 to 0.25.
    # external_fetch penalty of -0.15 brings final confidence to ~0.10.
    expected_base = 0.25
    external_penalty = -0.15
    expected_final = max(0.0, round(expected_base + external_penalty, 6))
    assert diagnostics.get("signal_integrity_scores", {}).get("Unified_integrity_score") == pytest.approx(
        expected_base, abs=1e-6,
    )
    assert "external_fetch_confidence_penalty" in reasons
    assert diagnostics.get("confidence_score") == pytest.approx(expected_final, abs=1e-6)
    assert payload["competitive_analysis"]["confidence"] == pytest.approx(expected_final, abs=1e-6)


@pytest.mark.parametrize(
    "prioritization",
    [
        {"insight_quality": "partial_insight", "uncertainty_mode": False},
        {"insight_quality": "full_insight", "uncertainty_mode": True},
    ],
)
def test_llm_node_degraded_modes_emit_schema_valid_withheld_recommendations(
    monkeypatch,
    prioritization: dict[str, object],
) -> None:
    monkeypatch.setattr("agent.nodes.llm_node.load_env_files", lambda: None)
    monkeypatch.setattr("agent.nodes.llm_node._build_adapter", lambda **_kw: object())
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
        "prioritization": prioritization,
    }

    result = llm_node(state)
    payload = json.loads(result["final_response"])

    # Regression guard: analyze/export routers re-validate final_response.
    validated = InsightOutput.model_validate(payload)
    strategic = validated.strategic_recommendations
    items = [
        strategic.immediate_actions[0],
        strategic.mid_term_moves[0],
        strategic.defensive_strategies[0],
        strategic.offensive_strategies[0],
    ]

    # Must stay unique across sections and carry explicit context terms.
    assert len(items) == len(set(items))
    context_terms = (
        "competitor",
        "gap",
        "strength",
        "weakness",
        "growth",
        "revenue",
        "retention",
        "risk",
    )
    for item in items:
        lower = item.lower()
        assert lower.startswith("conditional:")
        assert any(term in lower for term in context_terms)


def test_llm_node_degraded_mode_uses_critical_news_fallback(monkeypatch) -> None:
    monkeypatch.setattr("agent.nodes.llm_node.load_env_files", lambda: None)
    monkeypatch.setattr("agent.nodes.llm_node._build_adapter", lambda **_kw: object())
    monkeypatch.setattr(
        "agent.nodes.llm_node.generate_with_retry",
        lambda *_args, **_kwargs: _synthetic_llm_output(),
    )

    state = {
        "business_type": "saas",
        "entity_name": "acme",
        "saas_kpi_data": success({"records": [{"computed_kpis": {"mrr": {"value": 100.0}}}]}),
        "risk_data": success({"risk_score": 42.0}),
        "competitive_context": {
            "available": True,
            "source": "external_fetch",
            "peer_count": 3,
            "peers": ["Slack", "Notion", "Asana"],
            "metrics": ["listed_price_usd_mean"],
            "benchmark_rows_count": 0,
            "numeric_signals": [],
            "news_highlights": [
                {
                    "competitor": "Slack",
                    "title": "Slack reports major outage and security incident",
                    "snippet": "Downtime and security disclosures increased enterprise risk concerns.",
                    "published_at": "2026-03-20T00:00:00+00:00",
                    "url": "https://example.com/slack-incident",
                    "criticality_score": 3,
                }
            ],
        },
        "prioritization": {"insight_quality": "full_insight", "uncertainty_mode": True},
    }

    result = llm_node(state)
    payload = json.loads(result["final_response"])
    validated = InsightOutput.model_validate(payload)

    immediate = validated.strategic_recommendations.immediate_actions[0].lower()
    assert "most critical competitor news signal" in immediate
    assert "slack" in immediate


def test_llm_node_self_analysis_only_ignores_competitor_context(monkeypatch) -> None:
    monkeypatch.setattr("agent.nodes.llm_node.load_env_files", lambda: None)
    monkeypatch.setattr("agent.nodes.llm_node._build_adapter", lambda **_kw: object())
    monkeypatch.setattr(
        "agent.nodes.llm_node.generate_with_retry",
        lambda *_args, **_kwargs: _synthetic_llm_output(),
    )

    state = {
        "business_type": "saas",
        "entity_name": "acme",
        "self_analysis_only": True,
        "saas_kpi_data": success({"records": [{"computed_kpis": {"mrr": {"value": 100.0}}}]}),
        "risk_data": success({"risk_score": 42.0}),
        "competitive_context": {
            "available": True,
            "source": "external_fetch",
            "peer_count": 3,
            "peers": ["Slack", "Notion", "Asana"],
            "metrics": ["listed_price_usd_mean"],
            "benchmark_rows_count": 0,
            "numeric_signals": [],
            "news_highlights": [
                {
                    "competitor": "Slack",
                    "title": "Slack reports outage",
                    "snippet": "Downtime risk event.",
                    "published_at": "2026-03-20T00:00:00+00:00",
                    "url": "https://example.com/slack-outage",
                    "criticality_score": 3,
                }
            ],
        },
        "prioritization": {"insight_quality": "full_insight", "uncertainty_mode": True},
    }

    result = llm_node(state)
    payload = json.loads(result["final_response"])
    validated = InsightOutput.model_validate(payload)
    immediate = validated.strategic_recommendations.immediate_actions[0].lower()

    assert "data-only improvement plan" in immediate
    assert "competitor news signal" not in immediate
