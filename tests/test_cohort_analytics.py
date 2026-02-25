from __future__ import annotations

from agent.nodes.node_result import status_of
from agent.nodes.prioritization_node import prioritization_node
from agent.nodes.risk_node import risk_node
from app.services.cohort_analytics import compute_cohort_analytics


def test_cohort_analytics_computes_retention_decay_lifetime_and_acceleration() -> None:
    rows = [
        {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "metric_name": "active_customer_count",
            "metric_value": 100.0,
            "signup_month": "2025-11",
            "acquisition_channel": "Paid",
            "segment": "SMB",
            "metadata_json": {},
        },
        {
            "timestamp": "2026-02-01T00:00:00+00:00",
            "metric_name": "active_customer_count",
            "metric_value": 90.0,
            "signup_month": "2025-11",
            "acquisition_channel": "Paid",
            "segment": "SMB",
            "metadata_json": {},
        },
        {
            "timestamp": "2026-03-01T00:00:00+00:00",
            "metric_name": "active_customer_count",
            "metric_value": 80.0,
            "signup_month": "2025-11",
            "acquisition_channel": "Paid",
            "segment": "SMB",
            "metadata_json": {},
        },
        {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "metric_name": "churned_customer_count",
            "metric_value": 5.0,
            "signup_month": "2025-11",
            "acquisition_channel": "Paid",
            "segment": "SMB",
            "metadata_json": {},
        },
        {
            "timestamp": "2026-02-01T00:00:00+00:00",
            "metric_name": "churned_customer_count",
            "metric_value": 9.0,
            "signup_month": "2025-11",
            "acquisition_channel": "Paid",
            "segment": "SMB",
            "metadata_json": {},
        },
        {
            "timestamp": "2026-03-01T00:00:00+00:00",
            "metric_name": "churned_customer_count",
            "metric_value": 14.0,
            "signup_month": "2025-11",
            "acquisition_channel": "Paid",
            "segment": "SMB",
            "metadata_json": {},
        },
    ]

    result = compute_cohort_analytics(
        rows,
        active_metric_names=("active_customer_count",),
        churn_metric_names=("churned_customer_count",),
    )
    signals = result["signals"]

    assert result["status"] == "success"
    assert signals["retention_decay"] is not None
    assert signals["lifetime_estimate"] is not None
    assert signals["churn_acceleration"] is not None
    assert signals["worst_cohort"]["cohort_key"] in {"signup_month", "acquisition_channel", "segment"}


def test_sparse_cohorts_degrade_to_partial_with_confidence_penalty() -> None:
    rows = [
        {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "metric_name": "active_customer_count",
            "metric_value": 100.0,
            "signup_month": "2025-11",
            "metadata_json": {},
        },
        {
            "timestamp": "2026-02-01T00:00:00+00:00",
            "metric_name": "active_customer_count",
            "metric_value": 95.0,
            "signup_month": "2025-11",
            "metadata_json": {},
        },
    ]

    result = compute_cohort_analytics(
        rows,
        active_metric_names=("active_customer_count",),
        churn_metric_names=("churned_customer_count",),
    )

    assert result["status"] == "partial"
    assert result["confidence_score"] < 1.0
    assert result["signals"]["sparse_cohorts"] >= 1
    assert len(result["warnings"]) > 0


def _kpi_payload_for_risk() -> dict:
    return {
        "records": [
            {
                "period_end": "2026-01-31T00:00:00+00:00",
                "created_at": "2026-01-31T00:00:00+00:00",
                "computed_kpis": {
                    "revenue": {"value": 100.0},
                    "churn_rate": {"value": 0.05},
                    "conversion_rate": {"value": 0.02},
                },
            },
            {
                "period_end": "2026-02-28T00:00:00+00:00",
                "created_at": "2026-02-28T00:00:00+00:00",
                "computed_kpis": {
                    "revenue": {"value": 95.0},
                    "churn_rate": {"value": 0.08},
                    "conversion_rate": {"value": 0.018},
                },
            },
        ]
    }


def test_risk_node_uses_cohort_signal_when_forecast_missing(monkeypatch) -> None:
    class _DummySession:
        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> bool:
            return False

        def commit(self) -> None:
            return None

    class _FakeRiskOrchestrator:
        def __init__(self, _session: object) -> None:
            self._session = _session

        def generate_risk_score(self, entity_name: str, kpi_data: dict, forecast_data: dict) -> dict:
            _ = (kpi_data, forecast_data)
            return {
                "entity_name": entity_name,
                "risk_score": 44,
                "risk_level": "moderate",
            }

    monkeypatch.setattr("agent.nodes.risk_node.SessionLocal", lambda: _DummySession())
    monkeypatch.setattr("agent.nodes.risk_node.RiskOrchestrator", _FakeRiskOrchestrator)

    state = {
        "business_type": "general_timeseries",
        "entity_name": "Acme",
        "kpi_data": {"status": "success", "payload": _kpi_payload_for_risk()},
        "forecast_data": None,
        "segmentation": {
            "status": "success",
            "payload": {
                "cohort_analytics": {
                    "status": "success",
                    "confidence_score": 0.95,
                    "warnings": [],
                    "signals": {
                        "churn_acceleration": 0.08,
                        "risk_hint": "high",
                    },
                }
            },
        },
    }

    updated = risk_node(state)
    risk_data = updated["risk_data"]
    assert status_of(risk_data) == "success"
    payload = risk_data["payload"]
    assert payload["cohort_signals_used"] is True
    assert payload["forecast_available"] is True
    assert payload["cohort_signal_status"] == "success"


def test_risk_node_success_with_warnings_when_cohort_signals_partial(monkeypatch) -> None:
    class _DummySession:
        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> bool:
            return False

        def commit(self) -> None:
            return None

    class _FakeRiskOrchestrator:
        def __init__(self, _session: object) -> None:
            self._session = _session

        def generate_risk_score(self, entity_name: str, kpi_data: dict, forecast_data: dict) -> dict:
            _ = (kpi_data, forecast_data)
            return {
                "entity_name": entity_name,
                "risk_score": 50,
                "risk_level": "moderate",
            }

    monkeypatch.setattr("agent.nodes.risk_node.SessionLocal", lambda: _DummySession())
    monkeypatch.setattr("agent.nodes.risk_node.RiskOrchestrator", _FakeRiskOrchestrator)

    state = {
        "business_type": "general_timeseries",
        "entity_name": "Acme",
        "kpi_data": {"status": "success", "payload": _kpi_payload_for_risk()},
        "forecast_data": None,
        "segmentation": {
            "status": "success",
            "payload": {
                "cohort_analytics": {
                    "status": "partial",
                    "confidence_score": 0.40,
                    "warnings": ["Sparse cohort history."],
                    "signals": {
                        "churn_acceleration": 0.08,
                        "risk_hint": "moderate",
                    },
                }
            },
        },
    }

    updated = risk_node(state)
    risk_data = updated["risk_data"]
    assert status_of(risk_data) == "success"
    assert abs(float(risk_data["confidence_score"]) - 0.40) < 1e-9
    assert any("partial cohort signals" in warning for warning in risk_data["warnings"])


def test_prioritization_node_uses_cohort_signals_for_focus() -> None:
    state = {
        "risk_data": {
            "status": "success",
            "payload": {"risk_score": 22.0, "risk_level": "low"},
        },
        "root_cause": {"status": "success", "payload": {}},
        "segmentation": {
            "status": "success",
            "payload": {
                "cohort_analytics": {
                    "status": "success",
                    "confidence_score": 0.90,
                    "signals": {
                        "risk_hint": "high",
                        "retention_decay": 0.12,
                        "churn_acceleration": 0.06,
                        "worst_cohort": {
                            "cohort_key": "segment",
                            "cohort_value": "SMB",
                        },
                    },
                }
            },
        },
    }

    updated = prioritization_node(state)
    prioritization = updated["prioritization"]
    assert prioritization["priority_level"] == "high"
    assert "segment=SMB" in prioritization["recommended_focus"]
    assert prioritization["cohort_signal_used"] is True
