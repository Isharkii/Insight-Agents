from __future__ import annotations

from agent.nodes.node_result import payload_of, status_of, success
from agent.nodes.unit_economics_node import unit_economics_node
from app.services.unit_economics import analyze_unit_economics


def _records_for_healthy_growth() -> list[dict]:
    return [
        {
            "period_end": "2026-01-31T00:00:00+00:00",
            "computed_kpis": {
                "ltv": {"value": 600.0},
                "cac": {"value": 200.0},
                "churn_rate": {"value": 0.04},
                "revenue": {"value": 10000.0},
                "active_customer_count": {"value": 100.0},
                "growth_rate": {"value": 0.03},
            },
        },
        {
            "period_end": "2026-02-28T00:00:00+00:00",
            "computed_kpis": {
                "ltv": {"value": 700.0},
                "cac": {"value": 210.0},
                "churn_rate": {"value": 0.035},
                "revenue": {"value": 11000.0},
                "active_customer_count": {"value": 105.0},
                "growth_rate": {"value": 0.04},
            },
        },
        {
            "period_end": "2026-03-31T00:00:00+00:00",
            "computed_kpis": {
                "ltv": {"value": 900.0},
                "cac": {"value": 220.0},
                "churn_rate": {"value": 0.03},
                "revenue": {"value": 12500.0},
                "active_customer_count": {"value": 110.0},
                "growth_rate": {"value": 0.05},
            },
        },
    ]


def _records_for_unsustainable_growth() -> list[dict]:
    return [
        {
            "period_end": "2026-01-31T00:00:00+00:00",
            "computed_kpis": {
                "ltv": {"value": 300.0},
                "cac": {"value": 250.0},
                "churn_rate": {"value": 0.10},
                "revenue": {"value": 10000.0},
                "active_customer_count": {"value": 100.0},
                "growth_rate": {"value": 0.04},
            },
        },
        {
            "period_end": "2026-02-28T00:00:00+00:00",
            "computed_kpis": {
                "ltv": {"value": 250.0},
                "cac": {"value": 300.0},
                "churn_rate": {"value": 0.13},
                "revenue": {"value": 10500.0},
                "active_customer_count": {"value": 100.0},
                "growth_rate": {"value": 0.05},
            },
        },
        {
            "period_end": "2026-03-31T00:00:00+00:00",
            "computed_kpis": {
                "ltv": {"value": 200.0},
                "cac": {"value": 320.0},
                "churn_rate": {"value": 0.16},
                "revenue": {"value": 11000.0},
                "active_customer_count": {"value": 100.0},
                "growth_rate": {"value": 0.06},
            },
        },
    ]


def test_analyze_unit_economics_computes_core_metrics_and_healthy_signals() -> None:
    result = analyze_unit_economics(_records_for_healthy_growth(), business_type="general_timeseries")

    assert result["status"] == "success"
    assert result["metrics"]["ltv"] == 900.0
    assert result["metrics"]["cac"] == 220.0
    assert result["metrics"]["ltv_cac_ratio"] == round(900.0 / 220.0, 6)
    assert result["metrics"]["churn_rate"] == 0.03
    assert result["metrics"]["revenue_per_customer"] == round(12500.0 / 110.0, 6)

    signal_names = {item["signal"] for item in result["signals"]}
    assert "healthy_growth" in signal_names
    assert "unit_economics_healthy" in signal_names
    assert "acquisition_inefficiency" not in signal_names


def test_analyze_unit_economics_flags_unsustainable_and_acquisition_inefficiency() -> None:
    result = analyze_unit_economics(
        _records_for_unsustainable_growth(),
        business_type="general_timeseries",
    )
    assert result["status"] == "success"

    signal_names = {item["signal"] for item in result["signals"]}
    assert "acquisition_inefficiency" in signal_names
    assert "unsustainable_growth" in signal_names
    assert "churn_crisis" in signal_names


def test_analyze_unit_economics_returns_insufficient_data_without_records() -> None:
    result = analyze_unit_economics([], business_type="general_timeseries")
    assert result["status"] == "insufficient_data"
    assert result["signal_summary"] == "data_insufficient"
    assert result["metrics"]["ltv_cac_ratio"] is None


def test_unit_economics_node_emits_success_envelope() -> None:
    state = {
        "business_type": "general_timeseries",
        "kpi_data": success({"records": _records_for_healthy_growth()}),
        "dataset_confidence": 1.0,
    }
    updated = unit_economics_node(state)
    envelope = updated["unit_economics_data"]
    assert status_of(envelope) == "success"

    payload = payload_of(envelope) or {}
    assert payload.get("status") == "success"
    assert payload.get("metrics", {}).get("ltv_cac_ratio") is not None


def test_unit_economics_node_skips_when_kpi_records_are_missing() -> None:
    state = {
        "business_type": "general_timeseries",
        "kpi_data": success({"records": []}),
    }
    updated = unit_economics_node(state)
    envelope = updated["unit_economics_data"]
    assert status_of(envelope) == "skipped"

