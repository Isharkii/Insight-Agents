from __future__ import annotations

from app.services.timeseries_factors import (
    TimeseriesFactorConfig,
    TimeseriesFactorsEngine,
    compute_timeseries_factors,
)


def test_noisy_series_flags_are_deterministic() -> None:
    noisy_series = [
        100, 108, 92, 107, 93, 106, 94, 105,
        95, 104, 96, 103, 97, 102, 98, 100,
    ]

    result = compute_timeseries_factors(noisy_series)

    assert result["momentum_up"] is False
    assert result["momentum_down"] is False
    assert result["volatility_regime"] == "low"
    assert result["structural_break_detected"] is False
    assert result["cycle_state"] == "neutral"


def test_trending_series_detects_upward_momentum() -> None:
    trending_series = [
        100, 102, 104, 106, 108, 110, 112, 114,
        116, 118, 120, 122, 124, 126, 128, 130,
    ]

    result = compute_timeseries_factors(trending_series)

    assert result["momentum_up"] is True
    assert result["momentum_down"] is False
    assert result["volatility_regime"] == "low"
    assert result["structural_break_detected"] is False
    assert result["cycle_state"] == "expansion"


def test_regime_shifted_series_detects_structural_break() -> None:
    regime_shifted_series = [
        100, 101, 99, 100, 102, 98, 101, 99, 100, 101, 99, 100,
        134, 136, 133, 137, 135, 136, 134, 137, 135, 136, 134, 135,
    ]

    engine = TimeseriesFactorsEngine(config=TimeseriesFactorConfig())
    result = engine.evaluate(regime_shifted_series)

    assert result["momentum_up"] is False
    assert result["momentum_down"] is False
    assert result["volatility_regime"] == "low"
    assert result["structural_break_detected"] is True
    assert result["cycle_state"] == "neutral"

