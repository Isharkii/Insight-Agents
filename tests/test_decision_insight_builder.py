"""Comprehensive tests for the decision insight digest builder."""

from __future__ import annotations

import json

import pytest

from agent.helpers.decision_insight_builder import (
    _compute_cv,
    _compute_trend_strength,
    _safe_float,
    assess_data_quality,
    build_insight_digest,
    compute_deltas,
    compute_kpi_insights,
    detect_anomalies_zscore,
    extract_drivers,
    summarize_forecasts,
)


_FULL_SERIES = {
    "mrr": [100.0, 105.0, 110.0, 108.0, 115.0, 120.0, 125.0, 130.0, 128.0, 135.0, 140.0, 145.0],
    "churn_rate": [0.05, 0.048, 0.052, 0.049, 0.045, 0.047, 0.043, 0.041, 0.044, 0.04, 0.038, 0.035],
    "arpu": [42.0, 43.5, 44.0, 43.0, 45.0, 46.0, 47.0, 48.5, 49.0, 50.0, 51.0, 52.0],
}


def _insight_map(series: dict[str, list[float | object]]) -> dict[str, object]:
    return {item.metric: item for item in compute_kpi_insights(series)}


def _delta_map(series: dict[str, list[float]]) -> dict[tuple[str, str], object]:
    return {(item.metric, item.period_label): item for item in compute_deltas(series)}


def test_build_digest_happy_path_contains_all_sections() -> None:
    digest = build_insight_digest(
        metric_series=_FULL_SERIES,
        forecast_payload={
            "forecasts": {
                "mrr": {
                    "forecast_data": {
                        "slope": 0.02,
                        "r_squared": 0.83,
                        "confidence_score": 0.8,
                        "datapoints_used": 12,
                    }
                }
            }
        },
        root_cause_payload={
            "contributing_factors": ["pricing pressure", {"factor": "slow onboarding", "impact": "-4.0"}],
            "root_causes": ["pricing pressure", "market saturation"],
        },
        role_contribution_payload={
            "top_contributors": [
                {"name": "Enterprise", "contribution_value": 12000, "contribution_pct": 0.62},
                {"name": "SMB", "contribution_value": -3000, "contribution_pct": -0.12},
            ]
        },
        risk_payload={
            "risk_score": "72.349",
            "risk_level": "high",
            "risk_categories": [
                {"name": "liquidity", "severity": "high", "score": "0.88"},
                {"category": "execution", "score": "bad"},
            ],
        },
        growth_payload={
            "primary_metric": "mrr",
            "primary_horizons": {
                "short_growth": "0.12",
                "mid_growth": 0.2,
                "long_growth": "bad",
                "cagr": 0.18,
            },
            "momentum": 0.7,
        },
        cohort_status="success",
        cohort_confidence=0.6,
        entity_name="ATLAS_DYNAMICS",
    )

    assert digest["entity_name"] == "ATLAS_DYNAMICS"
    assert 0.0 < digest["overall_confidence"] <= 1.0
    assert len(digest["kpi_insights"]) == 3
    assert len(digest["drivers"]) >= 4
    assert len(digest["forecast_summary"]) == 1
    assert isinstance(digest["deltas"], list)
    assert isinstance(digest["anomalies"], list)
    assert digest["data_quality"]["missing_components"] == []

    assert digest["risk_summary"]["status"] == "available"
    assert digest["risk_summary"]["risk_score"] == 72.35
    assert digest["risk_summary"]["risk_categories"][1]["score"] is None

    assert digest["growth_summary"]["status"] == "available"
    assert digest["growth_summary"]["primary_metric"] == "mrr"
    assert digest["growth_summary"]["short_growth"] == 0.12
    assert digest["growth_summary"]["momentum"] == 0.7
    assert "long_growth" not in digest["growth_summary"]


def test_build_digest_default_marks_cohort_as_missing() -> None:
    digest = build_insight_digest(metric_series=_FULL_SERIES)
    assert "cohort" in digest["data_quality"]["missing_components"]


def test_compute_kpi_insights_threshold_boundaries() -> None:
    by_metric = _insight_map(
        {
            "grow": [100.0, 106.0],
            "decline": [100.0, 94.0],
            "stable": [100.0, 104.0],
        }
    )
    assert by_metric["grow"].status == "growing"
    assert by_metric["decline"].status == "declining"
    assert by_metric["stable"].status == "stable"


def test_compute_kpi_insights_single_point_and_missing_values() -> None:
    by_metric = _insight_map(
        {
            "single": [42.0],
            "mixed": [100.0, "101.5", "bad", None, float("inf"), float("nan"), object(), 102.0],
        }
    )

    single = by_metric["single"]
    assert single.status == "unknown"
    assert single.previous_value is None
    assert single.change_pct is None
    assert single.volatility_label == "unknown"

    mixed = by_metric["mixed"]
    assert mixed.data_points == 3
    assert mixed.latest_value == 102.0
    assert mixed.previous_value == 101.5


def test_compute_kpi_insights_empty_series_marks_unknown() -> None:
    insight = compute_kpi_insights({"empty": []})[0]
    assert insight.status == "unknown"
    assert insight.data_points == 0
    assert "No valid data points" in insight.warnings


def test_high_volatility_reduces_confidence_and_adds_warning() -> None:
    volatile = compute_kpi_insights({"x": [100.0, 200.0, 50.0, 220.0, 40.0, 240.0, 30.0, 260.0]})[0]
    stable = compute_kpi_insights({"x": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]})[0]

    assert volatile.volatility_label == "high"
    assert any("High volatility" in warning for warning in volatile.warnings)
    assert volatile.confidence < stable.confidence


def test_detect_anomalies_spike_dip_and_serialization() -> None:
    series = {
        "spike_metric": [100.0, 101.0, 102.0, 100.0, 99.0, 250.0, 101.0, 100.0, 102.0, 101.0],
        "dip_metric": [100.0, 99.0, 101.0, 100.0, 102.0, 10.0, 100.0, 101.0, 99.0, 100.0],
    }
    anomalies = detect_anomalies_zscore(series, threshold=2.0)
    assert any(item.metric == "spike_metric" and item.anomaly_type == "spike" for item in anomalies)
    assert any(item.metric == "dip_metric" and item.anomaly_type == "dip" for item in anomalies)

    sample = anomalies[0].to_dict()
    assert set(sample.keys()) == {"metric", "index", "value", "z_score", "type", "magnitude"}


def test_detect_anomalies_skips_short_series() -> None:
    assert detect_anomalies_zscore({"short": [1.0, 2.0, 3.0, 4.0]}) == []


def test_compute_deltas_outputs_mom_and_qoq() -> None:
    deltas = _delta_map({"mrr": [100.0, 110.0, 120.0, 130.0, 150.0]})
    assert deltas[("mrr", "MoM")].absolute_change == 20.0
    assert deltas[("mrr", "QoQ")].absolute_change == 40.0
    assert deltas[("mrr", "MoM")].pct_change == pytest.approx(20.0 / 130.0)


def test_compute_deltas_handles_zero_previous_value() -> None:
    deltas = _delta_map({"x": [0.0, 5.0]})
    assert deltas[("x", "MoM")].pct_change is None


def test_extract_drivers_merges_sources_and_deduplicates_root_cause() -> None:
    drivers = extract_drivers(
        {
            "contributing_factors": ["onboarding friction", {"factor": "pricing", "impact": "-11", "direction": "negative"}],
            "root_causes": ["pricing", "retention gap"],
        },
        {
            "top_contributors": [
                {"name": "Enterprise", "contribution_value": 4000, "contribution_pct": 0.4},
                {"name": "SMB", "contribution_value": -1200, "contribution_pct": -0.2},
                "skip_me",
            ]
        },
    )

    names = [item.name for item in drivers]
    assert names.count("pricing") == 1
    assert "onboarding friction" in names
    assert "retention gap" in names

    pricing = next(item for item in drivers if item.name == "pricing")
    assert pricing.contribution_value == -11.0
    assert pricing.direction == "negative"

    enterprise = next(item for item in drivers if item.name == "Enterprise")
    smb = next(item for item in drivers if item.name == "SMB")
    assert enterprise.direction == "positive"
    assert smb.direction == "negative"


def test_extract_drivers_with_empty_payloads_returns_empty_list() -> None:
    assert extract_drivers(None, None) == []


def test_summarize_forecasts_skips_invalid_shapes() -> None:
    summaries = summarize_forecasts(
        {
            "forecasts": {
                "not_a_dict": 123,
                "missing_forecast_data": {},
                "bad_forecast_data": {"forecast_data": "oops"},
            }
        }
    )
    assert summaries == []


def test_summarize_forecasts_requires_forecasts_dict() -> None:
    assert summarize_forecasts({"forecasts": []}) == []


def test_summarize_forecasts_regression_fallback_directions_and_confidence() -> None:
    summaries = summarize_forecasts(
        {
            "forecasts": {
                "up": {
                    "forecast_data": {
                        "slope": "0.02",
                        "regression": {"r_squared": "0.91"},
                        "confidence_score": "0.7",
                        "datapoints_used": 10,
                    }
                },
                "down": {
                    "forecast_data": {
                        "slope": -0.02,
                        "r_squared": 0.5,
                        "confidence_score": "bad",
                        "datapoints_used": 8,
                    }
                },
                "unknown": {
                    "forecast_data": {
                        "slope": None,
                        "r_squared": 0.5,
                        "confidence_score": 0.4,
                        "datapoints_used": 8,
                    }
                },
                "moderate": {
                    "forecast_data": {
                        "slope": 0.0,
                        "r_squared": 0.35,
                        "confidence_score": 0.3,
                        "datapoints_used": 9,
                    }
                },
            }
        }
    )
    by_metric = {item.metric: item for item in summaries}

    assert by_metric["up"].direction == "upward"
    assert by_metric["up"].r_squared == 0.91
    assert by_metric["up"].confidence == 0.7

    assert by_metric["down"].direction == "downward"
    assert by_metric["down"].confidence == 0.0

    assert by_metric["unknown"].direction == "unknown"
    assert by_metric["moderate"].direction == "flat"
    assert by_metric["moderate"].is_valid is True
    assert "moderate" in by_metric["moderate"].validity_reason


def test_summarize_forecasts_validity_reason_priority() -> None:
    low_r2 = summarize_forecasts(
        {
            "forecasts": {
                "mrr": {
                    "forecast_data": {
                        "slope": 0.01,
                        "r_squared": 0.12,
                        "confidence_score": 0.2,
                        "datapoints_used": 8,
                    }
                }
            }
        }
    )[0]
    assert low_r2.is_valid is False
    assert "too low" in low_r2.validity_reason

    low_r2_and_low_dp = summarize_forecasts(
        {
            "forecasts": {
                "mrr": {
                    "forecast_data": {
                        "slope": 0.01,
                        "r_squared": 0.12,
                        "confidence_score": 0.2,
                        "datapoints_used": 3,
                    }
                }
            }
        }
    )[0]
    assert low_r2_and_low_dp.is_valid is False
    assert "datapoints" in low_r2_and_low_dp.validity_reason


def test_assess_data_quality_marks_component_states_and_missing() -> None:
    components, missing = assess_data_quality(
        kpi_status="success",
        kpi_confidence=0.9,
        kpi_data_points=20,
        forecast_status="insufficient_data",
        forecast_confidence=0.35,
        cohort_status="skipped",
        cohort_confidence=0.0,
        risk_status="failed",
        risk_confidence=0.0,
        growth_status="success",
        growth_confidence=0.7,
    )

    by_component = {item.component: item for item in components}
    assert by_component["kpi"].status == "available"
    assert by_component["forecast"].status == "partial"
    assert by_component["cohort"].status == "unavailable"
    assert by_component["risk"].status == "unavailable"
    assert by_component["growth"].status == "available"
    assert set(missing) == {"cohort", "risk"}


def test_build_digest_overall_confidence_uses_authority_weights() -> None:
    digest = build_insight_digest(
        metric_series={"x": [100.0, 110.0]},
        kpi_confidence=1.0,
        growth_status="success",
        growth_confidence=0.5,
        risk_status="failed",
        risk_confidence=0.0,
        forecast_status="insufficient_data",
        forecast_confidence=0.25,
        cohort_status="skipped",
        cohort_confidence=0.0,
    )

    # Available components: kpi(1.0), growth(0.8), forecast(0.4)
    expected = round((1.0 * 1.0 + 0.5 * 0.8 + 0.25 * 0.4) / (1.0 + 0.8 + 0.4), 4)
    assert digest["overall_confidence"] == expected
    assert digest["data_quality"]["overall_confidence"] == expected


def test_unavailable_risk_and_growth_payloads_are_marked_unavailable() -> None:
    digest = build_insight_digest(metric_series={"x": [1.0, 2.0]}, risk_payload=None, growth_payload=None)
    assert digest["risk_summary"] == {"status": "unavailable"}
    assert digest["growth_summary"] == {"status": "unavailable"}


def test_growth_momentum_string_is_preserved() -> None:
    digest = build_insight_digest(
        metric_series={"x": [1.0, 2.0]},
        growth_payload={"momentum": "accelerating"},
        cohort_status="success",
    )
    assert digest["growth_summary"]["momentum"] == "accelerating"


def test_internal_helpers_cover_edge_cases() -> None:
    assert _safe_float(None) is None
    assert _safe_float("bad") is None
    assert _safe_float(float("inf")) is None
    assert _safe_float("1.25") == 1.25

    assert _compute_trend_strength([1.0]) is None
    assert _compute_trend_strength([-1.0, 1.0]) == 2.0

    assert _compute_cv([1.0]) is None
    assert _compute_cv([1.0, -1.0]) == 0.0


def test_digest_is_json_serializable_without_nan_or_infinity() -> None:
    digest = build_insight_digest(
        metric_series={"x": [1.0, "2.0", float("nan"), float("inf"), "bad", 3.0]},
        forecast_payload={
            "forecasts": {
                "x": {
                    "forecast_data": {
                        "slope": None,
                        "r_squared": None,
                        "confidence_score": "bad",
                        "datapoints_used": 5,
                    }
                }
            }
        },
    )
    encoded = json.dumps(digest, allow_nan=False)
    assert "NaN" not in encoded
    assert "Infinity" not in encoded
