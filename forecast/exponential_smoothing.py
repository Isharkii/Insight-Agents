"""
forecast/exponential_smoothing.py

Deterministic exponential smoothing utilities for short-horizon forecasts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Any

_ZERO_GUARD = 1e-9


@dataclass(frozen=True)
class SmoothingConfig:
    horizon: int = 3
    parameter_grid: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)


def holt_winters_forecast(
    values: list[float],
    *,
    season_length: int | None = None,
    config: SmoothingConfig | None = None,
) -> dict[str, Any]:
    """
    Select and run deterministic exponential smoothing.

    Uses additive Holt-Winters when season_length is provided and sufficient
    history exists, otherwise Holt's linear trend method.
    """
    cfg = config or SmoothingConfig()
    n = len(values)
    if n < 3:
        return {
            "status": "insufficient_data",
            "model": "none",
            "forecast": {"month_1": None, "month_2": None, "month_3": None},
            "fitted_values": [],
            "residuals": [],
            "slope": None,
            "mae": None,
            "sse": None,
            "params": {},
            "warnings": ["Need at least 3 points for exponential smoothing."],
        }

    if season_length is not None and season_length >= 2 and n >= season_length * 2:
        return _fit_holt_winters_additive(
            values,
            season_length=season_length,
            horizon=cfg.horizon,
            grid=cfg.parameter_grid,
        )
    return _fit_holt_linear(
        values,
        horizon=cfg.horizon,
        grid=cfg.parameter_grid,
    )


def _fit_holt_linear(
    values: list[float],
    *,
    horizon: int,
    grid: tuple[float, ...],
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for alpha in grid:
        for beta in grid:
            fitted, residuals, level, trend = _run_holt_linear(
                values,
                alpha=alpha,
                beta=beta,
            )
            sse = sum(err * err for err in residuals)
            mae = sum(abs(err) for err in residuals) / max(1, len(residuals))
            if best is None or sse < float(best["sse"]):
                best = {
                    "alpha": alpha,
                    "beta": beta,
                    "fitted": fitted,
                    "residuals": residuals,
                    "level": level,
                    "trend": trend,
                    "sse": sse,
                    "mae": mae,
                }

    assert best is not None
    forecast_values = {
        f"month_{step}": round(best["level"] + (step * best["trend"]), 6)
        for step in range(1, horizon + 1)
    }
    return {
        "status": "ok",
        "model": "holt_linear",
        "forecast": forecast_values,
        "fitted_values": [round(v, 6) for v in best["fitted"]],
        "residuals": [round(v, 6) for v in best["residuals"]],
        "slope": round(best["trend"], 6),
        "mae": round(float(best["mae"]), 6),
        "sse": round(float(best["sse"]), 6),
        "params": {
            "alpha": round(float(best["alpha"]), 4),
            "beta": round(float(best["beta"]), 4),
            "gamma": None,
        },
        "warnings": [],
    }


def _fit_holt_winters_additive(
    values: list[float],
    *,
    season_length: int,
    horizon: int,
    grid: tuple[float, ...],
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for alpha in grid:
        for beta in grid:
            for gamma in grid:
                fitted, residuals, level, trend, seasonals = _run_holt_winters_additive(
                    values,
                    season_length=season_length,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                )
                sse = sum(err * err for err in residuals)
                mae = sum(abs(err) for err in residuals) / max(1, len(residuals))
                if best is None or sse < float(best["sse"]):
                    best = {
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                        "fitted": fitted,
                        "residuals": residuals,
                        "level": level,
                        "trend": trend,
                        "seasonals": seasonals,
                        "sse": sse,
                        "mae": mae,
                    }

    assert best is not None
    n = len(values)
    seasonals = best["seasonals"]
    forecast_values: dict[str, float] = {}
    for step in range(1, horizon + 1):
        seasonal_index = (n + step - 1) % season_length
        seasonal_component = seasonals[seasonal_index]
        forecast_values[f"month_{step}"] = round(
            best["level"] + (step * best["trend"]) + seasonal_component,
            6,
        )
    return {
        "status": "ok",
        "model": "holt_winters_additive",
        "forecast": forecast_values,
        "fitted_values": [round(v, 6) for v in best["fitted"]],
        "residuals": [round(v, 6) for v in best["residuals"]],
        "slope": round(best["trend"], 6),
        "mae": round(float(best["mae"]), 6),
        "sse": round(float(best["sse"]), 6),
        "season_length": season_length,
        "params": {
            "alpha": round(float(best["alpha"]), 4),
            "beta": round(float(best["beta"]), 4),
            "gamma": round(float(best["gamma"]), 4),
        },
        "warnings": [],
    }


def _run_holt_linear(
    values: list[float],
    *,
    alpha: float,
    beta: float,
) -> tuple[list[float], list[float], float, float]:
    level = float(values[0])
    trend = float(values[1] - values[0]) if len(values) >= 2 else 0.0
    fitted: list[float] = [level]
    residuals: list[float] = [0.0]

    for actual in values[1:]:
        prediction = level + trend
        fitted.append(prediction)
        residuals.append(float(actual - prediction))

        prev_level = level
        level = alpha * float(actual) + (1.0 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1.0 - beta) * trend
    return fitted, residuals, level, trend


def _run_holt_winters_additive(
    values: list[float],
    *,
    season_length: int,
    alpha: float,
    beta: float,
    gamma: float,
) -> tuple[list[float], list[float], float, float, list[float]]:
    n = len(values)
    seasonals = _initial_seasonals(values, season_length=season_length)

    level = mean(values[:season_length])
    trend = _initial_trend(values, season_length=season_length)

    fitted: list[float] = []
    residuals: list[float] = []
    for idx in range(n):
        seasonal = seasonals[idx % season_length]
        prediction = level + trend + seasonal
        fitted.append(prediction)
        residuals.append(float(values[idx] - prediction))

        prev_level = level
        level = alpha * (values[idx] - seasonal) + (1.0 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1.0 - beta) * trend
        seasonals[idx % season_length] = (
            gamma * (values[idx] - level)
            + (1.0 - gamma) * seasonal
        )
    return fitted, residuals, level, trend, seasonals


def _initial_trend(values: list[float], *, season_length: int) -> float:
    if len(values) < season_length * 2:
        return 0.0
    first = values[:season_length]
    second = values[season_length : season_length * 2]
    return (mean(second) - mean(first)) / max(float(season_length), _ZERO_GUARD)


def _initial_seasonals(values: list[float], *, season_length: int) -> list[float]:
    n_seasons = max(1, len(values) // season_length)
    season_averages: list[float] = []
    for season_idx in range(n_seasons):
        start = season_idx * season_length
        end = start + season_length
        chunk = values[start:end]
        if len(chunk) < season_length:
            break
        season_averages.append(mean(chunk))

    if not season_averages:
        return [0.0 for _ in range(season_length)]

    seasonals = [0.0 for _ in range(season_length)]
    counts = [0 for _ in range(season_length)]

    for season_idx, season_avg in enumerate(season_averages):
        start = season_idx * season_length
        for pos in range(season_length):
            idx = start + pos
            if idx >= len(values):
                break
            seasonals[pos] += values[idx] - season_avg
            counts[pos] += 1

    for pos in range(season_length):
        if counts[pos] > 0:
            seasonals[pos] /= float(counts[pos])
    return seasonals

