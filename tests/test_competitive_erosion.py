from __future__ import annotations

from app.services.statistics.competitive_erosion import simulate_competitive_erosion


def test_lower_competitive_score_produces_higher_erosion_impact() -> None:
    weak = simulate_competitive_erosion(
        current_market_share=0.30,
        competitive_score=25.0,
        industry_growth_rate=0.08,
        erosion_factor=0.06,
        horizon_periods=8,
        severity="moderate",
        decay_mode="linear",
    )
    strong = simulate_competitive_erosion(
        current_market_share=0.30,
        competitive_score=85.0,
        industry_growth_rate=0.08,
        erosion_factor=0.06,
        horizon_periods=8,
        severity="moderate",
        decay_mode="linear",
    )

    assert float(weak["erosion_impact_pct"]) > float(strong["erosion_impact_pct"])
    assert weak["market_share_series"][-1] < strong["market_share_series"][-1]


def test_severity_levels_scale_erosion_mild_to_severe() -> None:
    mild = simulate_competitive_erosion(
        current_market_share=0.25,
        competitive_score=40.0,
        industry_growth_rate=0.10,
        erosion_factor=0.05,
        horizon_periods=6,
        severity="mild",
    )
    moderate = simulate_competitive_erosion(
        current_market_share=0.25,
        competitive_score=40.0,
        industry_growth_rate=0.10,
        erosion_factor=0.05,
        horizon_periods=6,
        severity="moderate",
    )
    severe = simulate_competitive_erosion(
        current_market_share=0.25,
        competitive_score=40.0,
        industry_growth_rate=0.10,
        erosion_factor=0.05,
        horizon_periods=6,
        severity="severe",
    )

    assert mild["erosion_impact_pct"] < moderate["erosion_impact_pct"] < severe["erosion_impact_pct"]
    assert mild["market_share_series"][-1] > moderate["market_share_series"][-1] > severe["market_share_series"][-1]


def test_linear_decay_erodes_growth_faster_than_exponential_for_same_inputs() -> None:
    linear = simulate_competitive_erosion(
        current_market_share=0.20,
        competitive_score=45.0,
        industry_growth_rate=0.12,
        erosion_factor=0.03,
        horizon_periods=10,
        severity="moderate",
        decay_mode="linear",
    )
    exponential = simulate_competitive_erosion(
        current_market_share=0.20,
        competitive_score=45.0,
        industry_growth_rate=0.12,
        erosion_factor=0.03,
        horizon_periods=10,
        severity="moderate",
        decay_mode="exponential",
    )

    linear_growth = [v for v in linear["adjusted_growth_rate_series"] if v is not None]
    exp_growth = [v for v in exponential["adjusted_growth_rate_series"] if v is not None]
    assert sum(linear_growth) < sum(exp_growth)
    assert linear["erosion_impact_pct"] > exponential["erosion_impact_pct"]


def test_seeded_optional_noise_is_reproducible() -> None:
    run_a = simulate_competitive_erosion(
        current_market_share=0.22,
        competitive_score=55.0,
        industry_growth_rate=0.09,
        erosion_factor=0.04,
        horizon_periods=7,
        severity="moderate",
        decay_mode="exponential",
        seed=123,
        noise_std=0.01,
    )
    run_b = simulate_competitive_erosion(
        current_market_share=0.22,
        competitive_score=55.0,
        industry_growth_rate=0.09,
        erosion_factor=0.04,
        horizon_periods=7,
        severity="moderate",
        decay_mode="exponential",
        seed=123,
        noise_std=0.01,
    )
    run_c = simulate_competitive_erosion(
        current_market_share=0.22,
        competitive_score=55.0,
        industry_growth_rate=0.09,
        erosion_factor=0.04,
        horizon_periods=7,
        severity="moderate",
        decay_mode="exponential",
        seed=124,
        noise_std=0.01,
    )

    assert run_a == run_b
    assert run_a != run_c


def test_no_seed_means_no_randomness_even_with_noise_std() -> None:
    without_seed_a = simulate_competitive_erosion(
        current_market_share=0.35,
        competitive_score=50.0,
        industry_growth_rate=0.07,
        erosion_factor=0.05,
        horizon_periods=6,
        severity="moderate",
        noise_std=0.02,
    )
    without_seed_b = simulate_competitive_erosion(
        current_market_share=0.35,
        competitive_score=50.0,
        industry_growth_rate=0.07,
        erosion_factor=0.05,
        horizon_periods=6,
        severity="moderate",
        noise_std=0.02,
    )
    deterministic_reference = simulate_competitive_erosion(
        current_market_share=0.35,
        competitive_score=50.0,
        industry_growth_rate=0.07,
        erosion_factor=0.05,
        horizon_periods=6,
        severity="moderate",
        noise_std=0.0,
    )

    assert without_seed_a == without_seed_b
    assert without_seed_a == deterministic_reference

