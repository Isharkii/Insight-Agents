import json

import requests

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
    class _Response:
        status_code = 200
        text = ""

        @staticmethod
        def json() -> dict:
            return {
                "insight": "Revenue dipped due to higher churn.",
                "evidence": "Churn increased 8% over baseline.",
                "impact": "ARR risk in upcoming quarter.",
                "recommended_action": "Launch retention program.",
                "priority": "high",
                "confidence_score": 0.91,
                "pipeline_status": "success",
                "diagnostics": None,
            }

    monkeypatch.setattr(streamlit_app.requests, "post", lambda *_, **__: _Response())

    response = streamlit_app.run_pipeline(data=None, prompt="analyze business")
    _assert_final_response_contract(response)


def test_pipeline_forced_failure_returns_final_insight_response(monkeypatch) -> None:
    def _raise(*_, **__):
        raise requests.RequestException("forced pipeline failure")

    monkeypatch.setattr(streamlit_app.requests, "post", _raise)

    response = streamlit_app.run_pipeline(data=None, prompt="analyze business")
    _assert_final_response_contract(response)
