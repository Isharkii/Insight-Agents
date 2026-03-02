from __future__ import annotations

import json
import importlib

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.services.bi_export_service import ExportResult
from db.session import get_db

bi_export_router = importlib.import_module("app.api.routers.bi_export_router")


class _FakeBIService:
    def export(self, *_args, **_kwargs) -> ExportResult:
        return ExportResult(
            rows=[{"entity_name": "acme", "metric_name": "revenue", "metric_value": 100.0}],
            fields=["entity_name", "metric_name", "metric_value"],
        )


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(bi_export_router.router)
    def _fake_db():
        yield object()
    app.dependency_overrides[get_db] = _fake_db
    app.dependency_overrides[bi_export_router.get_bi_export_service] = lambda: _FakeBIService()
    return app


def test_export_powerbi_alias_returns_flat_json() -> None:
    client = TestClient(_build_app())
    response = client.get(
        "/export/powerbi",
        params={"dataset": "records", "format": "json"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset"] == "records"
    assert payload["rows"] == 1
    assert isinstance(payload["data"], list)
    assert payload["data"][0]["metric_name"] == "revenue"


def test_export_report_returns_insight_and_derived_signals(monkeypatch) -> None:
    class _FakeGraph:
        @staticmethod
        def invoke(_state):
            return {
                "final_response": json.dumps(
                    {
                        "competitive_analysis": {
                            "summary": "Synthetic competitor summary.",
                            "market_position": "Synthetic challenger market position.",
                            "relative_performance": "Synthetic growth metric trails competitor benchmark.",
                            "key_advantages": ["Synthetic ARPU advantage versus competitor median."],
                            "key_vulnerabilities": ["Synthetic churn weakness versus competitor benchmark."],
                            "confidence": 0.8,
                        },
                        "strategic_recommendations": {
                            "immediate_actions": ["Address synthetic competitor churn gap immediately."],
                            "mid_term_moves": ["Close synthetic growth gap versus competitor benchmark."],
                            "defensive_strategies": ["Defend against synthetic competitor retention strength."],
                            "offensive_strategies": ["Exploit synthetic competitor weakness in ARPU benchmark."],
                        },
                    }
                ),
                "growth_data": {"status": "success", "payload": {"primary_metric": "revenue"}},
                "timeseries_factors_data": {"status": "success", "payload": {"factors": {}}},
                "cohort_data": {"status": "skipped", "payload": {"reason": "cohort_not_applicable"}},
                "category_formula_data": {"status": "success", "payload": {"category": "general_timeseries"}},
                "multivariate_scenario_data": {
                    "status": "success",
                    "payload": {"scenario_simulation": {"scenarios": {}}},
                },
                "segmentation": {"status": "success", "payload": {"top_contributors": []}},
                "risk_data": {"status": "success", "payload": {"risk_score": 40}},
                "prioritization": {"priority_level": "medium"},
            }

    monkeypatch.setattr(bi_export_router, "insight_graph", _FakeGraph())
    client = TestClient(_build_app())
    response = client.get(
        "/export/report",
        params={"entity_name": "acme", "format": "json"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["entity_name"] == "acme"
    assert payload["insight_payload"]["competitive_analysis"]["summary"] == "Synthetic competitor summary."
    assert "derived_signals" in payload
    assert "risk" in payload["derived_signals"]
