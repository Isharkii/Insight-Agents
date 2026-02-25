from __future__ import annotations

from app.services.statistics.macro_normalizer import (
    align_macro_series_to_periods,
    compute_growth_vs_gdp_delta,
    compute_rate_adjusted_efficiency,
    compute_real_growth_rate,
    normalize_business_metrics_for_macro,
    normalize_rate_series,
)


def test_real_growth_is_lower_when_inflation_is_higher() -> None:
    nominal = normalize_rate_series([5.0, 5.0])  # 5%
    inflation = normalize_rate_series([10.0, 1.0])  # 10% vs 1%

    real = compute_real_growth_rate(nominal, inflation)
    assert round(float(real[0]), 6) == -0.045455
    assert round(float(real[1]), 6) == 0.039604
    assert real[0] < real[1]


def test_growth_vs_gdp_and_rate_adjusted_efficiency_formulas() -> None:
    real_growth = [0.04, 0.02]
    gdp_growth = [0.03, 0.03]
    policy_rate = [0.05, 0.01]

    delta = compute_growth_vs_gdp_delta(real_growth, gdp_growth)
    efficiency = compute_rate_adjusted_efficiency(real_growth, policy_rate)

    assert [round(float(v), 6) for v in delta] == [0.01, -0.01]
    assert [round(float(v), 6) for v in efficiency] == [0.038095, 0.019802]


def test_time_alignment_matches_kpi_periods_to_macro_period_ranges() -> None:
    kpi_periods = [
        "2026-01-31",
        "2026-02-28",
        "2026-03-31",
        "2026-04-30",
    ]
    gdp_rows = [
        {"period_start": "2026-01-01", "period_end": "2026-03-31", "value": 0.02},
        {"period_start": "2026-04-01", "period_end": "2026-06-30", "value": 0.03},
    ]

    aligned = align_macro_series_to_periods(kpi_periods, gdp_rows)
    assert [round(float(v), 6) for v in aligned] == [0.02, 0.02, 0.02, 0.03]


def test_end_to_end_macro_normalization_is_deterministic() -> None:
    payload = normalize_business_metrics_for_macro(
        kpi_period_ends=["2026-01-31", "2026-02-28", "2026-03-31"],
        nominal_growth_rates=[5.0, 5.0, 5.0],
        inflation_rows=[
            {"period_start": "2026-01-01", "period_end": "2026-01-31", "value": 10.0},
            {"period_start": "2026-02-01", "period_end": "2026-02-28", "value": 1.0},
            {"period_start": "2026-03-01", "period_end": "2026-03-31", "value": 1.0},
        ],
        gdp_rows=[
            {"period_start": "2026-01-01", "period_end": "2026-03-31", "value": 2.0},
        ],
        policy_rate_rows=[
            {"period_start": "2026-01-01", "period_end": "2026-03-31", "value": 5.0},
        ],
    )
    second = normalize_business_metrics_for_macro(
        kpi_period_ends=["2026-01-31", "2026-02-28", "2026-03-31"],
        nominal_growth_rates=[5.0, 5.0, 5.0],
        inflation_rows=[
            {"period_start": "2026-01-01", "period_end": "2026-01-31", "value": 10.0},
            {"period_start": "2026-02-01", "period_end": "2026-02-28", "value": 1.0},
            {"period_start": "2026-03-01", "period_end": "2026-03-31", "value": 1.0},
        ],
        gdp_rows=[
            {"period_start": "2026-01-01", "period_end": "2026-03-31", "value": 2.0},
        ],
        policy_rate_rows=[
            {"period_start": "2026-01-01", "period_end": "2026-03-31", "value": 5.0},
        ],
    )

    assert payload == second
    assert payload["formulas"]["real_growth_rate"] == "((1 + nominal_growth_rate) / (1 + inflation_rate)) - 1"
    assert payload["real_growth_rate"][0] == -0.045455
    assert payload["growth_vs_gdp_delta"][0] == -0.065455

