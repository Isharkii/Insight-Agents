from __future__ import annotations

import time

from app.services.decision_engine_service import DecisionEngineService
from forecast.robust_forecast import MIN_POINTS_REGRESSION


def test_safe_forecast_rejects_series_below_reliable_threshold() -> None:
    service = DecisionEngineService()

    payload = service._safe_forecast([100.0, 108.0, 114.0, 120.0, 127.0])

    assert payload["status"] == "insufficient_data"
    assert payload["forecast_available"] is False
    assert payload["rejection_reason_code"] == "below_reliable_datapoint_threshold"
    assert payload["minimum_required"] == MIN_POINTS_REGRESSION
    assert payload["guardrail"]["accepted"] is False


def test_safe_forecast_accepts_robust_series_with_guardrail_metadata() -> None:
    service = DecisionEngineService()
    values = [100.0, 106.0, 112.0, 119.0, 126.0, 133.0, 141.0, 150.0, 160.0, 171.0, 183.0, 196.0]

    payload = service._safe_forecast(values)

    assert payload["status"] == "ok"
    assert payload["forecast_available"] is True
    assert payload["guardrail"]["accepted"] is True
    assert payload["confidence_score"] <= payload["raw_confidence_score"]


def test_propagate_confidence_reduces_forecast_weight_for_high_volatility() -> None:
    series_confidence = {"confidence_score": 0.60, "components": [], "warnings": []}
    low_vol_forecast = {
        "status": "ok",
        "forecast_available": True,
        "confidence_score": 0.90,
        "volatility_regime": "low",
    }
    high_vol_forecast = {
        "status": "ok",
        "forecast_available": True,
        "confidence_score": 0.90,
        "volatility_regime": "high",
    }

    low_score, low_breakdown = DecisionEngineService._propagate_confidence(
        series_confidence=series_confidence,
        forecast=low_vol_forecast,
        anomaly_detected=False,
        churn_rate=0.0,
    )
    high_score, high_breakdown = DecisionEngineService._propagate_confidence(
        series_confidence=series_confidence,
        forecast=high_vol_forecast,
        anomaly_detected=False,
        churn_rate=0.0,
    )

    assert low_breakdown["weights"]["forecast"] > high_breakdown["weights"]["forecast"]
    assert low_score > high_score


def test_analyze_business_exposes_async_module_execution_metadata() -> None:
    service = DecisionEngineService()
    result = service.analyze_business(
        entity_name="Acme",
        business_type="saas",
        question="How healthy is growth?",
        context={
            "revenue_series": [100.0, 110.0, 125.0, 140.0],
            "churn_rate": 0.03,
            "conversion_rate": 0.04,
            "customers": 40,
            "ltv": 1200.0,
            "cac": 300.0,
        },
    )

    insight = result["pipeline"]["insight"]
    module_execution = insight["module_execution"]
    assert "kpi_extraction" in module_execution
    assert "trend_analysis" in module_execution
    assert "anomaly_detection" in module_execution
    assert "forecasting" in module_execution
    assert "unit_economics" in module_execution
    assert "execution_time_ms" in module_execution["forecasting"]
    assert "unit_economics" in insight


def test_analyze_business_forecast_timeout_is_isolated(monkeypatch) -> None:
    service = DecisionEngineService(module_timeouts={"forecasting": 0.01})

    def _slow_forecast(values):
        _ = values
        time.sleep(0.05)
        return {"status": "ok", "forecast_available": True, "confidence_score": 0.8}

    monkeypatch.setattr(service, "_safe_forecast", _slow_forecast)
    result = service.analyze_business(
        entity_name="Acme",
        business_type="saas",
        question="How healthy is growth?",
        context={
            "revenue_series": [100.0, 110.0, 120.0, 130.0],
            "churn_rate": 0.03,
            "conversion_rate": 0.04,
            "customers": 40,
        },
    )

    insight = result["pipeline"]["insight"]
    assert insight["forecast"]["status"] == "insufficient_data"
    assert insight["module_execution"]["forecasting"]["status"] == "timeout"
    assert "trend" in insight
    assert "kpis" in insight


def test_analyze_business_trend_error_is_isolated(monkeypatch) -> None:
    service = DecisionEngineService()

    def _boom(values):
        _ = values
        raise RuntimeError("trend module failed")

    monkeypatch.setattr(service, "_trend_analysis_module", _boom)
    result = service.analyze_business(
        entity_name="Acme",
        business_type="saas",
        question="How healthy is growth?",
        context={
            "revenue_series": [100.0, 108.0, 116.0, 125.0],
            "churn_rate": 0.03,
            "conversion_rate": 0.04,
            "customers": 40,
        },
    )

    insight = result["pipeline"]["insight"]
    assert insight["module_execution"]["trend_analysis"]["status"] == "error"
    assert insight["forecast"]["status"] in {"ok", "insufficient_data"}
    assert "trend_analysis_degraded" in insight["signals"]
