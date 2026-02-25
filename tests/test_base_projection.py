from __future__ import annotations

from app.services.statistics.base_projection import (
    BaseProjectionConfig,
    align_growth_to_periods,
    build_projection_periods,
    cagr_growth,
    project_base_case,
    rolling_average_growth,
)


def test_cagr_growth_formula() -> None:
    revenue = [100.0, 110.0, 121.0, 133.1]
    growth = cagr_growth(revenue)
    assert growth is not None
    assert round(float(growth), 6) == 0.1


def test_rolling_average_growth_formula() -> None:
    revenue = [100.0, 110.0, 99.0, 108.9, 119.79]
    growth = rolling_average_growth(revenue, window=3)
    assert growth is not None
    assert round(float(growth), 6) == 0.033333


def test_build_projection_periods_from_historical_quarters() -> None:
    periods = build_projection_periods(
        horizon_quarters=2,
        historical_period_ends=[
            "2025-03-31",
            "2025-06-30",
            "2025-09-30",
            "2025-12-31",
        ],
    )
    assert [str(item) for item in periods] == ["2026-03-31", "2026-06-30"]


def test_align_growth_to_periods_by_interval_containment() -> None:
    aligned = align_growth_to_periods(
        ["2026-03-31", "2026-06-30"],
        growth_rows=[
            {"period_start": "2026-01-01", "period_end": "2026-03-31", "value": 0.05},
            {"period_start": "2026-04-01", "period_end": "2026-06-30", "value": 0.06},
        ],
    )
    assert [round(float(v), 6) for v in aligned] == [0.05, 0.06]


def test_project_base_case_cagr_with_macro_anchors() -> None:
    result = project_base_case(
        [100.0, 110.0, 121.0, 133.1],
        industry_growth_rate=0.08,
        gdp_growth_rate=0.04,
        historical_period_ends=["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"],
        config=BaseProjectionConfig(method="cagr", horizon_quarters=2),
    )

    assert result["method"] == "cagr"
    assert result["projected_period_end"] == ["2026-03-31", "2026-06-30"]
    assert result["projected_growth_rate"] == [0.088, 0.088]
    assert result["projected_revenue"] == [144.8128, 157.556326]


def test_project_base_case_rolling_average_without_macro_data() -> None:
    cfg = BaseProjectionConfig(method="rolling_avg", horizon_quarters=2, rolling_window=3)
    result = project_base_case(
        [100.0, 110.0, 99.0, 108.9, 119.79],
        historical_period_ends=["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31"],
        config=cfg,
    )
    assert result["method"] == "rolling_avg"
    assert result["projected_growth_rate"] == [0.033333, 0.033333]
    assert result["projected_revenue"] == [123.783, 127.9091]


def test_project_base_case_is_deterministic() -> None:
    kwargs = {
        "historical_revenue": [100.0, 110.0, 121.0, 133.1],
        "industry_growth_rows": [
            {"period_start": "2026-01-01", "period_end": "2026-03-31", "value": 0.05},
            {"period_start": "2026-04-01", "period_end": "2026-06-30", "value": 0.06},
        ],
        "gdp_growth_rate": 0.02,
        "historical_period_ends": ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"],
        "config": BaseProjectionConfig(
            method="cagr",
            horizon_quarters=2,
            client_weight=0.0,
            industry_weight=1.0,
            gdp_weight=0.0,
        ),
    }
    first = project_base_case(**kwargs)
    second = project_base_case(**kwargs)
    assert first == second
    assert first["projected_growth_rate"] == [0.05, 0.06]
