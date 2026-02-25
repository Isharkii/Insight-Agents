from __future__ import annotations

from app.services.statistics.recession_modeling import (
    compute_revenue_shock_multiplier,
    extract_macro_shock_inputs,
    model_recession_projection,
)


def test_revenue_shock_multiplier_formula() -> None:
    multiplier = compute_revenue_shock_multiplier(
        gdp_contraction_rate=0.04,
        interest_rate_spike=0.02,
        industry_sensitivity_coefficient=1.5,
        gdp_weight=0.65,
        interest_weight=0.35,
        min_multiplier=0.35,
        max_multiplier=1.0,
    )
    assert round(float(multiplier), 6) == 0.9505


def test_recovery_curves_v_u_l_have_expected_shapes() -> None:
    base = [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0]
    common = {
        "base_projected_revenue": base,
        "gdp_contraction_rate": 0.06,
        "interest_rate_spike": 0.02,
        "industry_sensitivity_coefficient": 1.8,
        "shock_duration_quarters": 2,
    }

    v_shape = model_recession_projection(recovery_curve="v_shape", **common)
    u_shape = model_recession_projection(recovery_curve="u_shape", **common)
    l_shape = model_recession_projection(recovery_curve="l_shape", **common)

    assert len(v_shape["shock_phase_projection"]) == 2
    assert len(v_shape["recovery_projection"]) == 6

    assert v_shape["recovery_projection"][-1] == base[-1]
    assert u_shape["recovery_projection"][-1] == base[-1]
    assert l_shape["recovery_projection"][-1] < base[-1]

    assert v_shape["recovery_projection"][0] > u_shape["recovery_projection"][0]
    assert l_shape["recovery_time_estimate"] == "not_recovered_within_horizon"


def test_macro_metrics_schema_rows_are_derived_when_inputs_missing() -> None:
    macro_rows = [
        {
            "country_code": "US",
            "metric_name": "gdp",
            "period_start": "2025-10-01",
            "period_end": "2025-12-31",
            "value": 100.0,
        },
        {
            "country_code": "US",
            "metric_name": "gdp",
            "period_start": "2026-01-01",
            "period_end": "2026-03-31",
            "value": 95.0,
        },
        {
            "country_code": "US",
            "metric_name": "policy_rate",
            "period_start": "2026-01-01",
            "period_end": "2026-01-31",
            "value": 3.0,
        },
        {
            "country_code": "US",
            "metric_name": "policy_rate",
            "period_start": "2026-02-01",
            "period_end": "2026-02-28",
            "value": 5.0,
        },
    ]

    derived = extract_macro_shock_inputs(macro_rows, country_code="US")
    assert derived["gdp_contraction_rate"] == 0.05
    assert derived["interest_rate_spike"] == 0.02

    modeled = model_recession_projection(
        base_projected_revenue=[120.0, 123.0, 126.0, 129.0, 132.0, 135.0],
        gdp_contraction_rate=None,
        interest_rate_spike=None,
        industry_sensitivity_coefficient=1.4,
        shock_duration_quarters=2,
        recovery_curve="u_shape",
        macro_metric_rows=macro_rows,
        country_code="US",
    )

    assert "shock_phase_projection" in modeled
    assert "recovery_projection" in modeled
    assert "net_revenue_impact" in modeled
    assert "recovery_time_estimate" in modeled
    assert modeled["revenue_shock_multiplier"] is not None


def test_recession_modeling_is_deterministic() -> None:
    kwargs = {
        "base_projected_revenue": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        "gdp_contraction_rate": 0.03,
        "interest_rate_spike": 0.01,
        "industry_sensitivity_coefficient": 1.2,
        "shock_duration_quarters": 2,
        "recovery_curve": "u_shape",
    }
    first = model_recession_projection(**kwargs)
    second = model_recession_projection(**kwargs)
    assert first == second

