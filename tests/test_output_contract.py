import json

import pytest
from pydantic import ValidationError

from llm_synthesis.schema import InsightOutput


def _payload() -> dict:
    return {
        "competitive_analysis": {
            "summary": "Competitor benchmark suggests churn pressure against peer retention metrics.",
            "market_position": "Current market position is challenger relative to peer benchmark share.",
            "relative_performance": "Growth and churn metrics underperform competitor benchmark medians.",
            "key_advantages": ["ARPU metric is stronger than competitor median."],
            "key_vulnerabilities": ["Churn metric is weaker than competitor benchmark."],
            "confidence": 0.87,
        },
        "strategic_recommendations": {
            "immediate_actions": ["Address competitor gap in churn weakness this quarter."],
            "mid_term_moves": ["Close growth metric gap versus competitor benchmark trends."],
            "defensive_strategies": ["Defend segments where competitor strength in retention is rising."],
            "offensive_strategies": ["Exploit competitor weakness in ARPU benchmark efficiency."],
        },
    }


def test_insight_output_contract() -> None:
    output = InsightOutput(**_payload())

    required = {
        "competitive_analysis",
        "strategic_recommendations",
    }
    assert set(output.model_dump().keys()) == required
    assert isinstance(output.competitive_analysis.confidence, float)

    serialized = output.model_dump_json()
    parsed = json.loads(serialized)
    assert parsed["competitive_analysis"]["summary"].startswith("Competitor benchmark")
    assert isinstance(parsed["competitive_analysis"]["confidence"], float)


def test_insight_output_rejects_extra_fields() -> None:
    data = _payload()
    data["extra_field"] = "not allowed"
    with pytest.raises(ValidationError):
        InsightOutput(**data)


def test_low_confidence_requires_conditional_recommendations() -> None:
    data = _payload()
    data["competitive_analysis"]["confidence"] = 0.4
    with pytest.raises(ValidationError):
        InsightOutput(**data)


def test_recommendations_reject_repetition() -> None:
    data = _payload()
    duplicate = "Address competitor churn gap in priority segment."
    data["strategic_recommendations"]["immediate_actions"] = [duplicate]
    data["strategic_recommendations"]["mid_term_moves"] = [duplicate]
    with pytest.raises(ValidationError):
        InsightOutput(**data)
