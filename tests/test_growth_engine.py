from __future__ import annotations

from agent.nodes.node_result import status_of
from agent.nodes.prioritization_node import prioritization_node
from agent.nodes.risk_node import risk_node
from app.services.statistics.growth_engine import (
    compute_growth_context,
    compute_growth_signals,
    metric_growth_config,
)


def test_growth_engine_computes_horizon_rates_cagr_and_acceleration() -> None:
    series = [100.0, 104.0, 108.0, 112.0, 116.0, 120.0, 126.0, 132.0, 138.0, 144.0]
    cfg = metric_growth_config("revenue")

    result = compute_growth_signals(series, metric_name="revenue", config=cfg)
    horizons = result["moving_growth_rates"]
    cagr = result["cagr"]
    accel = result["acceleration_metrics"]

    assert horizons["short"] is not None
    assert horizons["mid"] is not None
    assert cagr["rate"] is not None
    assert accel["short_to_mid"] is not None
    assert accel["mid_to_long"] is not None
    assert accel["trend_acceleration"] is not None


def test_growth_engine_sets_explicit_insufficient_history_flags() -> None:
    short_series = [100.0, 102.0, 103.0]
    result = compute_growth_signals(short_series, metric_name="timeseries_value")

    assert result["status"] == "partial"
    insuff = result["insufficient_history"]
    assert insuff["short"] is True
    assert insuff["mid"] is True
    assert insuff["long"] is True
    assert insuff["cagr"] is True
    assert insuff["acceleration"] is True


def test_growth_context_selects_primary_metric_by_candidates() -> None:
    metric_series = {
        "timeseries_value": [100.0, 105.0, 110.0, 120.0, 130.0],
        "churn_rate": [0.04, 0.05, 0.06, 0.07, 0.08],
    }
    context = compute_growth_context(
        metric_series,
        preferred_metric_candidates=("timeseries_value",),
    )

    assert context["primary_metric"] == "timeseries_value"
    assert "primary_horizons" in context
    assert "short_growth" in context["primary_horizons"]


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
                    "revenue": {"value": 98.0},
                    "churn_rate": {"value": 0.07},
                    "conversion_rate": {"value": 0.018},
                },
            },
        ]
    }


def test_risk_node_uses_growth_horizons_when_forecast_missing(monkeypatch) -> None:
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
            assert abs(float(kpi_data["revenue_growth_delta"]) - (-0.08)) < 1e-9
            assert abs(float(forecast_data["slope"]) - (-0.05)) < 1e-9
            assert abs(float(forecast_data["deviation_percentage"]) - 0.05) < 1e-9
            return {
                "entity_name": entity_name,
                "risk_score": 55,
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
                "growth_context": {
                    "status": "success",
                    "confidence_score": 0.95,
                    "warnings": [],
                    "primary_horizons": {
                        "short_growth": -0.08,
                        "mid_growth": -0.05,
                        "long_growth": -0.03,
                        "trend_acceleration": -0.02,
                        "insufficient_history": {
                            "short": False,
                            "mid": False,
                            "long": False,
                            "cagr": False,
                            "acceleration": False,
                        },
                    },
                }
            },
        },
    }

    updated = risk_node(state)
    risk_data = updated["risk_data"]
    assert status_of(risk_data) == "success"
    payload = risk_data["payload"]
    assert payload["growth_signals_used"] is True
    assert payload["growth_signal_status"] == "success"


def test_prioritization_uses_growth_horizon_severity_and_focus() -> None:
    state = {
        "risk_data": {
            "status": "success",
            "payload": {"risk_score": 20.0, "risk_level": "low"},
        },
        "root_cause": {"status": "success", "payload": {}},
        "segmentation": {
            "status": "success",
            "payload": {
                "growth_context": {
                    "status": "success",
                    "confidence_score": 0.9,
                    "primary_horizons": {
                        "short_growth": -0.12,
                        "mid_growth": -0.08,
                        "long_growth": -0.04,
                        "trend_acceleration": -0.03,
                        "insufficient_history": {
                            "short": False,
                            "mid": False,
                            "long": False,
                            "cagr": False,
                            "acceleration": False,
                        },
                    },
                }
            },
        },
    }

    updated = prioritization_node(state)
    prioritization = updated["prioritization"]
    assert prioritization["priority_level"] == "high"
    assert "growth decline" in prioritization["recommended_focus"]
    assert prioritization["growth_signal_used"] is True

