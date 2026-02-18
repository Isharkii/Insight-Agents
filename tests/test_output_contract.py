import json

import pytest
from pydantic import ValidationError

from llm_synthesis.schema import InsightOutput


def _payload() -> dict:
    return {
        "insight": "Revenue decline due to churn.",
        "evidence": "Churn increased in mid-market segment.",
        "impact": "ARR pressure in next quarter.",
        "recommended_action": "Launch retention campaign.",
        "priority": "High",
        "confidence_score": 0.87,
    }


def test_insight_output_contract() -> None:
    output = InsightOutput(**_payload())

    required = {
        "insight",
        "evidence",
        "impact",
        "recommended_action",
        "priority",
        "confidence_score",
    }
    assert set(output.model_dump().keys()) == required
    assert isinstance(output.confidence_score, float)

    serialized = output.model_dump_json()
    parsed = json.loads(serialized)
    assert parsed["insight"] == "Revenue decline due to churn."
    assert isinstance(parsed["confidence_score"], float)


def test_insight_output_rejects_extra_fields() -> None:
    data = _payload()
    data["extra_field"] = "not allowed"
    with pytest.raises(ValidationError):
        InsightOutput(**data)
