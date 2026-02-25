from __future__ import annotations

import math

from app.services.statistics.inflation_adjustment import (
    align_cpi_to_kpi_periods,
    build_inflation_adjusted_series,
    compound_inflation_rates,
    compute_cpi_inflation_rate,
    compute_inflation_adjusted_revenue,
    compute_real_growth_rate,
)


def test_compute_real_growth_rate_matches_formula() -> None:
    real = compute_real_growth_rate([5.0], [10.0])
    assert round(float(real[0]), 6) == -0.045455

    real_low_inflation = compute_real_growth_rate([5.0], [1.0])
    assert round(float(real_low_inflation[0]), 6) == 0.039604


def test_align_cpi_to_kpi_periods_supports_quarterly_and_monthly_windows() -> None:
    kpi_period_ends = ["2026-01-31", "2026-02-28", "2026-03-31", "2026-04-30"]
    cpi_rows = [
        {"period_start": "2026-01-01", "period_end": "2026-03-31", "cpi_index": 250.0},
        {"period_start": "2026-04-01", "period_end": "2026-06-30", "cpi_index": 252.0},
    ]

    aligned = align_cpi_to_kpi_periods(kpi_period_ends, cpi_rows)
    assert [round(float(value), 6) for value in aligned] == [250.0, 250.0, 250.0, 252.0]


def test_compute_inflation_adjusted_revenue_handles_missing_cpi_gracefully() -> None:
    revenue = [1000.0, 1100.0, 1210.0, 1300.0]
    cpi_index = [200.0, None, 210.0, None]

    adjusted = compute_inflation_adjusted_revenue(
        revenue,
        cpi_index,
        base_cpi=200.0,
        missing_policy="ffill",
    )
    assert [round(float(value), 6) for value in adjusted] == [1000.0, 1100.0, 1152.380952, 1238.095238]

    strict = compute_inflation_adjusted_revenue(
        revenue,
        cpi_index,
        base_cpi=200.0,
        missing_policy="nan",
    )
    assert round(float(strict[0]), 6) == 1000.0
    assert math.isnan(float(strict[1]))
    assert round(float(strict[2]), 6) == 1152.380952


def test_compute_cpi_inflation_rate_and_compounding() -> None:
    cpi_index = [200.0, 202.0, 204.02]
    rates = compute_cpi_inflation_rate(cpi_index)
    assert rates[0] != rates[0]  # NaN
    assert [round(float(rates[1]), 6), round(float(rates[2]), 6)] == [0.01, 0.01]

    compounded = compound_inflation_rates([1.0, 1.0, 1.0])
    assert [round(float(value), 6) for value in compounded] == [0.01, 0.0201, 0.030301]


def test_build_inflation_adjusted_series_is_deterministic() -> None:
    kwargs = {
        "kpi_period_ends": ["2026-01-31", "2026-02-28", "2026-03-31"],
        "nominal_growth_rate": [5.0, 5.0, 5.0],
        "revenue": [1000.0, 1050.0, 1102.5],
        "cpi_rows": [
            {"period_start": "2026-01-01", "period_end": "2026-01-31", "cpi_index": 200.0},
            {"period_start": "2026-02-01", "period_end": "2026-02-28", "cpi_index": 202.0},
            {"period_start": "2026-03-01", "period_end": "2026-03-31", "cpi_index": 204.02},
        ],
    }
    first = build_inflation_adjusted_series(**kwargs)
    second = build_inflation_adjusted_series(**kwargs)

    assert first == second
    assert first["formulas"]["real_growth_rate"] == "((1 + nominal_growth) / (1 + inflation_rate)) - 1"
    assert first["real_growth_rate"][1] == 0.039604

