from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routers.decision_engine_router import router


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def test_analyze_business_sync_flow_success(monkeypatch) -> None:
    monkeypatch.setenv("DECISION_ENGINE_API_KEY", "test-key")
    client = TestClient(_build_app())

    response = client.post(
        "/analyze_business",
        headers={"X-API-Key": "test-key", "X-Request-ID": "req-123"},
        json={
            "entity_name": "Acme",
            "business_type": "saas",
            "question": "How healthy is growth?",
            "context": {
                "revenue_series": [100.0, 120.0, 135.0],
                "churn_rate": 0.04,
                "conversion_rate": 0.05,
                "customers": 50,
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "req-123"
    assert body["success"] is True
    assert body["error"] is None

    data = body["data"]
    assert data["entity_name"] == "Acme"
    assert data["pipeline"]["data"]["status"] == "ok"
    assert "insight" in data["pipeline"]
    assert "reasoning" in data["pipeline"]
    assert "strategy" in data["pipeline"]
    assert isinstance(data["signals_generated"], list)


def test_analyze_business_rejects_invalid_api_key(monkeypatch) -> None:
    monkeypatch.setenv("DECISION_ENGINE_API_KEY", "test-key")
    client = TestClient(_build_app())

    response = client.post(
        "/analyze_business",
        headers={"X-API-Key": "wrong-key"},
        json={
            "entity_name": "Acme",
            "business_type": "saas",
            "question": "How healthy is growth?",
            "context": {},
        },
    )
    assert response.status_code == 401


def test_analyze_metrics_sync_flow_success(monkeypatch) -> None:
    monkeypatch.setenv("DECISION_ENGINE_API_KEY", "test-key")
    client = TestClient(_build_app())

    response = client.post(
        "/analyze_metrics",
        headers={"X-API-Key": "test-key"},
        json={
            "entity_name": "Acme",
            "period": "monthly",
            "metrics": [
                {"name": "revenue", "values": [100.0, 110.0, 130.0]},
                {"name": "churn_rate", "values": [0.03, 0.04, 0.05]},
            ],
        },
    )
    assert response.status_code == 200

    body = response.json()
    assert body["success"] is True
    assert body["error"] is None
    assert body["data"]["entity_name"] == "Acme"
    assert len(body["data"]["metrics_analyzed"]) == 2
    assert isinstance(body["data"]["signals_generated"], list)


def test_system_health_reports_service_status(monkeypatch) -> None:
    monkeypatch.setenv("DECISION_ENGINE_API_KEY", "test-key")
    client = TestClient(_build_app())
    response = client.get("/system_health")
    assert response.status_code == 200

    body = response.json()
    assert body["success"] is True
    assert body["data"]["status"] == "ok"
    assert body["data"]["checks"]["decision_engine_service"] == "ok"


def test_analyze_business_returns_insufficient_data_for_sparse_forecast(monkeypatch) -> None:
    monkeypatch.setenv("DECISION_ENGINE_API_KEY", "test-key")
    client = TestClient(_build_app())

    response = client.post(
        "/analyze_business",
        headers={"X-API-Key": "test-key"},
        json={
            "entity_name": "Acme",
            "business_type": "saas",
            "question": "Forecast this business",
            "context": {
                "revenue_series": [100.0],
                "churn_rate": 0.03,
                "conversion_rate": 0.04,
                "customers": 40,
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()["data"]["pipeline"]["insight"]
    assert payload["forecast"]["status"] == "insufficient_data"
    assert payload["forecast"]["forecast_available"] is False
    assert "forecast_insufficient_data" in payload["signals"]


def test_analyze_business_rejects_two_point_series_forecast(monkeypatch) -> None:
    monkeypatch.setenv("DECISION_ENGINE_API_KEY", "test-key")
    client = TestClient(_build_app())

    response = client.post(
        "/analyze_business",
        headers={"X-API-Key": "test-key"},
        json={
            "entity_name": "Acme",
            "business_type": "saas",
            "question": "Forecast this business",
            "context": {
                "revenue_series": [100.0, 110.0],
                "churn_rate": 0.03,
                "conversion_rate": 0.04,
                "customers": 40,
            },
        },
    )

    assert response.status_code == 200
    forecast = response.json()["data"]["pipeline"]["insight"]["forecast"]
    assert forecast["status"] == "insufficient_data"
    assert forecast["forecast_available"] is False


def test_analyze_metrics_marks_sparse_series_as_insufficient(monkeypatch) -> None:
    monkeypatch.setenv("DECISION_ENGINE_API_KEY", "test-key")
    client = TestClient(_build_app())

    response = client.post(
        "/analyze_metrics",
        headers={"X-API-Key": "test-key"},
        json={
            "entity_name": "Acme",
            "period": "monthly",
            "metrics": [
                {"name": "revenue", "values": [100.0]},
                {"name": "conversion_rate", "values": [0.03, 0.04, 0.05]},
            ],
        },
    )
    assert response.status_code == 200
    metrics = response.json()["data"]["metrics_analyzed"]
    revenue = next(item for item in metrics if item["metric_name"] == "revenue")
    assert revenue["status"] == "insufficient_data"
    assert revenue["forecast"]["forecast_available"] is False
