import json

import streamlit_app
from llm_synthesis.schema import FinalInsightResponse


_REQUIRED_KEYS = frozenset(FinalInsightResponse.model_fields.keys())
_VALID_PRIORITIES = {"low", "medium", "high", "critical"}


def _assert_final_response_contract(response: object) -> None:
    assert isinstance(response, FinalInsightResponse)
    assert not isinstance(response, str)
    assert not isinstance(response, dict)

    payload = response.model_dump()
    assert set(payload.keys()) == _REQUIRED_KEYS
    assert isinstance(response.confidence_score, float)
    assert response.priority in _VALID_PRIORITIES


def test_pipeline_success_returns_final_insight_response(monkeypatch) -> None:
    class _GraphSuccess:
        def invoke(self, _: dict) -> dict:
            return {
                "final_response": json.dumps(
                    {
                        "insight": "Revenue dipped due to higher churn.",
                        "evidence": "Churn increased 8% over baseline.",
                        "impact": "ARR risk in upcoming quarter.",
                        "recommended_action": "Launch retention program.",
                        "priority": "high",
                        "confidence_score": 0.91,
                    }
                )
            }

    monkeypatch.setattr(
        streamlit_app,
        "_load_backend_handles",
        lambda: {
            "graph": _GraphSuccess(),
            "intent_node": lambda _: {},
            "csv_service": object(),
            "session_factory": object(),
        },
    )

    response = streamlit_app.run_pipeline(data=None, prompt="analyze business")
    _assert_final_response_contract(response)


def test_pipeline_forced_failure_returns_final_insight_response(monkeypatch) -> None:
    class _GraphFailure:
        def invoke(self, _: dict) -> dict:
            raise RuntimeError("forced pipeline failure")

    monkeypatch.setattr(
        streamlit_app,
        "_load_backend_handles",
        lambda: {
            "graph": _GraphFailure(),
            "intent_node": lambda _: {},
            "csv_service": object(),
            "session_factory": object(),
        },
    )

    response = streamlit_app.run_pipeline(data=None, prompt="analyze business")
    _assert_final_response_contract(response)
