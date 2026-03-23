from __future__ import annotations

import json

from agent.nodes.llm_node import llm_node
from agent.nodes.node_result import success
from agent.nodes.prioritization_node import prioritization_node
from agent.nodes.signal_conflict_node import signal_conflict_node
from llm_synthesis.schema import InsightOutput


def _synthetic_llm_output_max_confidence() -> InsightOutput:
    payload = {
        "competitive_analysis": {
            "summary": "Market growth momentum is positive while revenue quality remains mixed.",
            "market_position": "Current market position is moderate with revenue volatility risk.",
            "relative_performance": "Relative performance shows stronger growth but higher churn rate risk.",
            "key_advantages": ["Higher revenue growth metric supports a short-term strength."],
            "key_vulnerabilities": ["Elevated churn vulnerability increases retention risk."],
            "confidence": 1.0,
        },
        "strategic_recommendations": {
            "immediate_actions": ["Reduce churn risk by targeting retention gaps this quarter."],
            "mid_term_moves": ["Optimize revenue growth quality by prioritizing high-retention segments."],
            "defensive_strategies": ["Address weakness in retention trend before scaling spend."],
            "offensive_strategies": ["Increase growth in segments showing revenue momentum strength."],
        },
    }
    return InsightOutput.model_validate(payload)


def test_signal_conflict_node_detects_global_conflicts() -> None:
    state = {
        "business_type": "general_timeseries",
        "entity_name": "Acme",
        "kpi_data": success(
            {
                "records": [
                    {
                        "period_end": "2026-01-31T00:00:00+00:00",
                        "computed_kpis": {
                            "revenue": {"value": 100.0},
                            "churn_rate": {"value": 0.05},
                            "conversion_rate": {"value": 0.02},
                        },
                    },
                    {
                        "period_end": "2026-02-28T00:00:00+00:00",
                        "computed_kpis": {
                            "revenue": {"value": 120.0},
                            "churn_rate": {"value": 0.08},
                            "conversion_rate": {"value": 0.03},
                        },
                    },
                ]
            }
        ),
        "growth_data": success(
            {
                "primary_horizons": {
                    "short_growth": 0.20,
                    "mid_growth": 0.12,
                    "long_growth": 0.10,
                    "trend_acceleration": -0.04,
                }
            }
        ),
        "cohort_data": success(
            {
                "signals": {
                    "retention_decay": 0.12,
                    "churn_acceleration": 0.06,
                }
            }
        ),
    }

    updated = signal_conflict_node(state)
    envelope = updated["signal_conflicts"]
    payload = envelope["payload"]
    conflict_result = payload["conflict_result"]

    assert envelope["status"] == "success"
    assert conflict_result["conflict_count"] > 0
    assert payload["confidence_adjustment"]["penalty_applied"] > 0.0
    assert any("Conflict:" in warning for warning in envelope["warnings"])


def test_prioritization_reflects_uncertainty_from_conflicts() -> None:
    state = {
        "risk_data": success({"risk_score": 22.0, "risk_level": "low"}),
        "root_cause": success({}),
        "segmentation": success({"growth_context": {"primary_horizons": {}}}),
        "signal_conflicts": success(
            {
                "conflict_result": {
                    "status": "conflicts_detected",
                    "conflict_count": 2,
                    "total_severity": 1.4,
                    "confidence_penalty": 0.2,
                    "uncertainty_flag": True,
                    "warnings": ["Conflict: revenue_growth_delta vs churn_delta"],
                }
            }
        ),
    }

    updated = prioritization_node(state)
    payload = updated["prioritization"]

    assert payload["signal_conflict_count"] == 2
    assert payload["strategy_uncertainty_flag"] is True
    # With total_severity > 1.0, uncertainty_mode activates and
    # recommendations are withheld instead of appended.
    assert "conflicting signals" in payload["recommended_focus"].lower()
    assert payload.get("uncertainty_mode") is True
    assert payload.get("decision") == "withheld"
    assert payload["confidence_score"] < payload["reasoning_confidence_score"]


def test_prioritization_keeps_actions_for_moderate_conflicts_without_uncertainty_flag() -> None:
    state = {
        "risk_data": success({"risk_score": 22.0, "risk_level": "low"}),
        "root_cause": success({}),
        "segmentation": success({"growth_context": {"primary_horizons": {}}}),
        "signal_conflicts": success(
            {
                "conflict_result": {
                    "status": "conflicts_detected",
                    "conflict_count": 2,
                    "total_severity": 1.4,
                    "confidence_penalty": 0.2,
                    "uncertainty_flag": False,
                    "warnings": ["Conflict: revenue_growth_delta vs churn_delta"],
                }
            }
        ),
    }

    updated = prioritization_node(state)
    payload = updated["prioritization"]

    assert payload["signal_conflict_count"] == 2
    assert payload["strategy_uncertainty_flag"] is False
    assert payload.get("uncertainty_mode") is False
    assert payload.get("decision") == "active"


def test_llm_node_applies_global_conflict_penalty_in_diagnostics(monkeypatch) -> None:
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
        "signal_conflicts": success(
            {
                "conflict_result": {
                    "status": "conflicts_detected",
                    "conflict_count": 1,
                    "total_severity": 1.0,
                    "confidence_penalty": 0.2,
                    "uncertainty_flag": True,
                    "warnings": ["Conflict: revenue_growth_delta vs churn_delta"],
                },
                "confidence_adjustment": {
                    "base_confidence": 0.7,
                    "adjusted_confidence": 0.5,
                },
            }
        ),
    }

    result = llm_node(state)
    diagnostics = result["envelope_diagnostics"]
    payload = json.loads(result["final_response"])

    reasons = {
        str(item.get("reason"))
        for item in diagnostics.get("confidence_adjustments", [])
        if isinstance(item, dict)
    }
    assert "global_signal_conflict_penalty" in reasons
    assert diagnostics.get("signal_conflicts", {}).get("conflict_count") == 1
    assert payload["competitive_analysis"]["confidence"] == diagnostics.get("confidence_score")
