import json

import requests

import streamlit_app
from llm_synthesis.schema import FinalInsightResponse


_REQUIRED_KEYS = frozenset(FinalInsightResponse.model_fields.keys())


def _assert_final_response_contract(response: object) -> None:
    assert isinstance(response, FinalInsightResponse)
    assert not isinstance(response, str)
    assert not isinstance(response, dict)

    payload = response.model_dump()
    assert set(payload.keys()) == _REQUIRED_KEYS
    assert isinstance(response.competitive_analysis.confidence, float)
    assert "competitive_analysis" in payload
    assert "strategic_recommendations" in payload


def test_pipeline_success_returns_final_insight_response(monkeypatch) -> None:
    class _Response:
        status_code = 200
        text = ""

        @staticmethod
        def json() -> dict:
            return {
                "competitive_analysis": {
                    "summary": "Competitor benchmark indicates retention pressure.",
                    "market_position": "Entity is challenger versus benchmark peers.",
                    "relative_performance": "Churn metric is weaker than competitor median.",
                    "key_advantages": ["ARPU metric remains stronger than competitor median."],
                    "key_vulnerabilities": ["Retention metric underperforms competitor benchmark."],
                    "confidence": 0.91,
                },
                "strategic_recommendations": {
                    "immediate_actions": ["Close competitor churn gap in highest-risk segment."],
                    "mid_term_moves": ["Improve growth-rate benchmark position versus peers."],
                    "defensive_strategies": ["Protect segments where competitor strength in churn is highest."],
                    "offensive_strategies": ["Target competitor weakness in ARPU conversion efficiency."],
                },
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
