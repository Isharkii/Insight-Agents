from __future__ import annotations

from agent.nodes.cohort_analytics_node import cohort_analytics_node
from agent.nodes.multivariate_scenario_node import multivariate_scenario_node
from agent.nodes.node_result import success
from agent.nodes.signal_conflict_node import signal_conflict_node
from agent.nodes.timeseries_factors_node import timeseries_factors_node
from app.services.statistics.multivariate import compute_multivariate_context
from app.services.statistics.seasonality import detect_seasonality
from forecast.robust_forecast import RobustForecast


def _monthly_records(metric_values: dict[str, list[float]]) -> list[dict]:
    rows: list[dict] = []
    points = max(len(values) for values in metric_values.values())
    for idx in range(points):
        month = (idx % 12) + 1
        year = 2025 + (idx // 12)
        computed: dict[str, dict[str, float]] = {}
        for metric_name, values in metric_values.items():
            if idx < len(values):
                computed[metric_name] = {"value": float(values[idx])}
        rows.append(
            {
                "period_end": f"{year:04d}-{month:02d}-01T00:00:00+00:00",
                "created_at": f"{year:04d}-{month:02d}-01T00:00:00+00:00",
                "computed_kpis": computed,
                "signup_month": "2025-10",
                "acquisition_channel": "Paid",
                "segment": "SMB",
                "metadata_json": {"signup_month": "2025-10", "channel": "Paid", "segment": "SMB"},
            }
        )
    return rows


def test_seasonality_detector_exposes_spectral_analysis_for_dynamic_periods() -> None:
    values = [10.0, 13.0, 12.0, 16.0, 11.0] * 8
    result = detect_seasonality(values)
    periods = {int(candidate.get("period")) for candidate in result.get("candidates", [])}

    assert "spectral_analysis" in result.get("diagnostics", {})
    assert result["diagnostics"]["spectral_analysis"].get("dominant_period") is not None
    assert 5 in periods


def test_multivariate_context_defaults_to_spearman_outputs() -> None:
    metric_series = {
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "y": [1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0],
    }
    context = compute_multivariate_context(metric_series)
    correlation = context["correlation"]
    pair = correlation["pairs"][0]

    assert correlation["method"] == "spearman"
    assert pair["spearman_correlation"] is not None
    assert correlation["spearman_matrix"]["x"]["y"] is not None


def test_robust_forecast_includes_model_selection_and_residual_diagnostics() -> None:
    values = [100.0, 120.0, 85.0, 115.0] * 6
    result = RobustForecast().forecast(values)
    diagnostics = result["diagnostics"]

    assert result["status"] == "ok"
    assert diagnostics["model_selection"]["selected_model"] == result["model"]
    assert "residual_diagnostics" in diagnostics
    assert diagnostics["residual_diagnostics"]["status"] in {"ok", "insufficient_data"}


def test_cohort_node_includes_survival_analysis_payload() -> None:
    records = _monthly_records(
        {
            "active_customer_count": [100.0, 92.0, 88.0, 82.0, 78.0],
            "churned_customer_count": [4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    state = {
        "business_type": "general_timeseries",
        "entity_name": "Acme",
        "kpi_data": success({"fetched_for": "Acme", "records": records}),
    }
    updated = cohort_analytics_node(state)
    payload = updated["cohort_data"]["payload"]

    assert "survival_analysis" in payload
    assert payload["survival_analysis"]["profiles_count"] >= 1


def test_timeseries_factors_node_includes_changepoint_output() -> None:
    records = _monthly_records(
        {
            "timeseries_value": [100.0] * 8 + [140.0] * 8,
        }
    )
    state = {
        "business_type": "general_timeseries",
        "entity_name": "Acme",
        "kpi_data": success({"fetched_for": "Acme", "records": records}),
        "growth_data": success({"primary_metric": "timeseries_value"}),
    }
    updated = timeseries_factors_node(state)
    factors = updated["timeseries_factors_data"]["payload"]["factors"]

    assert "changepoints" in factors
    assert "summary" in factors["changepoints"]


def test_signal_conflict_node_surfaces_leading_indicator_metadata() -> None:
    records = _monthly_records(
        {
            "revenue": [100.0, 105.0, 112.0, 118.0, 123.0, 130.0, 138.0, 146.0],
            "churn_rate": [0.03, 0.032, 0.034, 0.037, 0.041, 0.045, 0.049, 0.054],
        }
    )
    state = {
        "business_type": "general_timeseries",
        "entity_name": "Acme",
        "kpi_data": success({"fetched_for": "Acme", "records": records}),
        "growth_data": success({"primary_horizons": {"short_growth": 0.2, "trend_acceleration": -0.05}}),
        "cohort_data": success({"signals": {"churn_acceleration": 0.04}}),
    }
    updated = signal_conflict_node(state)
    payload = updated["signal_conflicts"]["payload"]

    assert "leading_indicators" in payload
    assert "temporal_conflicts" in payload


def test_multivariate_scenario_node_includes_causal_inference_output() -> None:
    records = _monthly_records(
        {
            "timeseries_value": [80.0, 85.0, 89.0, 94.0, 99.0, 105.0, 111.0, 118.0, 124.0, 131.0, 139.0, 148.0],
            "active_customer_count": [40.0, 42.0, 44.0, 47.0, 49.0, 52.0, 55.0, 58.0, 61.0, 65.0, 69.0, 73.0],
        }
    )
    state = {
        "business_type": "general_timeseries",
        "entity_name": "Acme",
        "kpi_data": success({"fetched_for": "Acme", "records": records}),
    }
    updated = multivariate_scenario_node(state)
    payload = updated["multivariate_scenario_data"]["payload"]
    causal = payload["causal_inference"]

    assert "summary" in causal
    assert causal["summary"]["pairs_tested"] >= 0

