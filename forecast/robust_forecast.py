"""
forecast/robust_forecast.py

Tiered deterministic forecasting with model selection and residual diagnostics.
"""

from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Any

from app.services.statistics.seasonality import detect_seasonality
from forecast.base import BaseForecastModel
from forecast.diagnostics import residual_diagnostics
from forecast.exponential_smoothing import holt_winters_forecast

# Tier thresholds
MIN_POINTS_INSUFFICIENT = 3
MIN_POINTS_REGRESSION = 6
MIN_POINTS_ROBUST = 12

_CONFIDENCE_CAP_MINIMAL = 0.40
_CONFIDENCE_CAP_STANDARD = 0.80

_FORECAST_HORIZON = 3
_ROLLING_WINDOW = 3
_ZERO_GUARD = 1e-9


def _rolling_average(values: list[float], *, window: int) -> list[float]:
    w = max(1, int(window))
    out: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - w + 1)
        bucket = values[start : idx + 1]
        out.append(round(mean(bucket), 6))
    return out


def _coefficient_of_variation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    sigma = pstdev(values)
    return sigma / max(abs(mu), _ZERO_GUARD)


def _period_returns(values: list[float]) -> list[float]:
    if len(values) < 2:
        return []
    return [
        (values[idx] - values[idx - 1]) / max(abs(values[idx - 1]), _ZERO_GUARD)
        for idx in range(1, len(values))
    ]


def _ols_regression(values: list[float]) -> dict[str, float]:
    n = len(values)
    x = list(range(n))
    mean_x = sum(x) / n
    mean_y = mean(values)

    cov_xy = sum((x[idx] - mean_x) * (values[idx] - mean_y) for idx in range(n))
    var_x = sum((x[idx] - mean_x) ** 2 for idx in range(n))
    if var_x < _ZERO_GUARD:
        return {
            "slope": 0.0,
            "intercept": mean_y,
            "r_squared": 0.0,
            "residual_std": 0.0,
        }

    slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x

    fitted = [(slope * x[idx]) + intercept for idx in range(n)]
    ss_tot = sum((values[idx] - mean_y) ** 2 for idx in range(n))
    ss_res = sum((values[idx] - fitted[idx]) ** 2 for idx in range(n))
    r_squared = max(0.0, 1.0 - (ss_res / max(ss_tot, _ZERO_GUARD)))
    residual_std = math.sqrt(ss_res / max(1, n - 2)) if n > 2 else 0.0

    return {
        "slope": round(slope, 6),
        "intercept": round(intercept, 6),
        "r_squared": round(r_squared, 6),
        "residual_std": round(residual_std, 6),
    }


def _ols_fitted_and_residuals(
    values: list[float],
    *,
    slope: float,
    intercept: float,
) -> tuple[list[float], list[float]]:
    fitted = [round((slope * idx) + intercept, 6) for idx in range(len(values))]
    residuals = [round(values[idx] - fitted[idx], 6) for idx in range(len(values))]
    return fitted, residuals


def _volatility_regime(values: list[float], *, window: int = 6) -> dict[str, Any]:
    returns = _period_returns(values)
    if len(returns) < 2:
        return {
            "regime": "insufficient_history",
            "recent_vol": None,
            "baseline_vol": None,
            "ratio": None,
        }

    w = min(window, len(returns))
    recent = returns[-w:]
    baseline = returns[:-w] if len(returns) > w else returns
    recent_vol = pstdev(recent) if len(recent) >= 2 else 0.0
    baseline_vol = pstdev(baseline) if len(baseline) >= 2 else recent_vol
    ratio = recent_vol / max(baseline_vol, _ZERO_GUARD)

    if ratio >= 1.25:
        regime = "high"
    elif ratio <= 0.80:
        regime = "low"
    else:
        regime = "normal"

    return {
        "regime": regime,
        "recent_vol": round(recent_vol, 6),
        "baseline_vol": round(baseline_vol, 6),
        "ratio": round(ratio, 6),
    }


def _depth_score(n: int) -> float:
    if n <= 1:
        return 0.0
    return min(1.0, math.log(n) / math.log(MIN_POINTS_ROBUST))


def _stability_score(cov: float) -> float:
    penalty = min(0.7, cov / 2.0)
    return max(0.3, round(1.0 - penalty, 6))


def _compute_confidence(
    *,
    n: int,
    r_squared: float | None,
    cov: float,
    tier: str,
) -> float:
    depth = _depth_score(n)
    fit = max(0.1, float(r_squared)) if r_squared is not None else 1.0
    stability = _stability_score(cov)
    raw = depth * fit * stability

    if tier == "minimal":
        cap = _CONFIDENCE_CAP_MINIMAL
    elif tier == "standard":
        cap = _CONFIDENCE_CAP_STANDARD
    else:
        cap = 1.0
    return round(max(0.0, min(cap, raw)), 6)


def _trend_slope_assessment(
    slope: float,
    average_value: float,
    r_squared: float,
) -> dict[str, Any]:
    normalised_slope = slope if average_value == 0.0 else slope / abs(average_value)

    if r_squared < 0.3:
        label = "inconclusive"
    elif normalised_slope > 0.05:
        label = "strong_uptrend"
    elif normalised_slope > 0.01:
        label = "uptrend"
    elif normalised_slope < -0.05:
        label = "strong_downtrend"
    elif normalised_slope < -0.01:
        label = "downtrend"
    else:
        label = "stable"

    return {
        "label": label,
        "normalised_slope": round(normalised_slope, 6),
        "r_squared": round(r_squared, 6),
        "statistically_meaningful": r_squared >= 0.3,
    }


def _forecast_values_from_slope(last_value: float, slope: float) -> dict[str, float]:
    return {
        f"month_{step}": round(last_value + (step * slope), 6)
        for step in range(1, _FORECAST_HORIZON + 1)
    }


def _safe_float(value: object, *, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return float(parsed)


class RobustForecast(BaseForecastModel):
    """
    Tiered deterministic forecast with model selection.

    0-2 points: insufficient
    3-5 points: rolling average extrapolation
    6+ points: OLS or exponential smoothing selection + diagnostics
    """

    def forecast(self, values: list[float]) -> dict:
        n = len(values)
        if n < MIN_POINTS_INSUFFICIENT:
            return self._insufficient_result(n)

        cov = _coefficient_of_variation(values)
        rolling = _rolling_average(values, window=_ROLLING_WINDOW)
        volatility = _volatility_regime(values)

        if n < MIN_POINTS_REGRESSION:
            return self._minimal_result(
                values=values,
                rolling=rolling,
                cov=cov,
                volatility=volatility,
            )

        tier = "robust" if n >= MIN_POINTS_ROBUST else "standard"
        return self._regression_result(
            values=values,
            rolling=rolling,
            cov=cov,
            volatility=volatility,
            tier=tier,
        )

    @staticmethod
    def _insufficient_result(n: int) -> dict[str, Any]:
        return {
            "tier": "insufficient",
            "status": "insufficient_data",
            "model": "none",
            "confidence_score": 0.0,
            "input_points": n,
            "minimum_required": MIN_POINTS_INSUFFICIENT,
            "data_quality": {
                "coefficient_of_variation": None,
                "volatility": {"regime": "insufficient_history"},
                "rolling_average": [],
            },
            "regression": None,
            "trend": None,
            "forecast": {"month_1": None, "month_2": None, "month_3": None},
            "deviation_percentage": None,
            "slope": None,
            "diagnostics": {
                "method": "none",
                "reason": (
                    f"Insufficient data: need at least {MIN_POINTS_INSUFFICIENT} "
                    f"points, got {n}."
                ),
            },
            "warnings": [
                f"Forecast requires at least {MIN_POINTS_INSUFFICIENT} points; got {n}.",
            ],
        }

    @staticmethod
    def _minimal_result(
        *,
        values: list[float],
        rolling: list[float],
        cov: float,
        volatility: dict[str, Any],
    ) -> dict[str, Any]:
        n = len(values)
        last_rolling = rolling[-1] if rolling else values[-1]
        forecast_values = {
            f"month_{step}": round(last_rolling, 6)
            for step in range(1, _FORECAST_HORIZON + 1)
        }
        confidence = _compute_confidence(n=n, r_squared=None, cov=cov, tier="minimal")

        return {
            "tier": "minimal",
            "status": "ok",
            "model": "rolling_average",
            "confidence_score": confidence,
            "input_points": n,
            "minimum_required": MIN_POINTS_INSUFFICIENT,
            "data_quality": {
                "coefficient_of_variation": round(cov, 6),
                "volatility": volatility,
                "rolling_average": rolling,
            },
            "regression": None,
            "trend": {
                "label": "inconclusive",
                "normalised_slope": None,
                "r_squared": None,
                "statistically_meaningful": False,
            },
            "forecast": forecast_values,
            "deviation_percentage": 0.0,
            "slope": None,
            "diagnostics": {
                "method": "rolling_average",
                "window": _ROLLING_WINDOW,
                "reason": (
                    f"Only {n} points available (need {MIN_POINTS_REGRESSION} for "
                    "regression); using rolling-average extrapolation."
                ),
            },
            "warnings": [
                f"Below regression threshold ({MIN_POINTS_REGRESSION}); "
                "using rolling-average extrapolation with capped confidence.",
            ],
        }

    @staticmethod
    def _regression_result(
        *,
        values: list[float],
        rolling: list[float],
        cov: float,
        volatility: dict[str, Any],
        tier: str,
    ) -> dict[str, Any]:
        n = len(values)
        avg = mean(values)
        reg = _ols_regression(values)
        ols_fitted, ols_residuals = _ols_fitted_and_residuals(
            values,
            slope=float(reg["slope"]),
            intercept=float(reg["intercept"]),
        )
        ols_mae = sum(abs(err) for err in ols_residuals) / max(1, len(ols_residuals))
        ols_forecast = _forecast_values_from_slope(values[-1], float(reg["slope"]))
        predicted_last = ols_fitted[-1] if ols_fitted else values[-1]
        ols_deviation = (
            (values[-1] - predicted_last) / max(abs(predicted_last), _ZERO_GUARD)
        )

        seasonality = detect_seasonality(values)
        seasonality_detected = bool(seasonality.get("detected"))
        seasonality_period = seasonality.get("primary_period")
        season_length: int | None = None
        if seasonality_detected and isinstance(seasonality_period, int):
            if seasonality_period >= 2:
                season_length = int(seasonality_period)

        smoothing = holt_winters_forecast(values, season_length=season_length)
        smoothing_ok = str(smoothing.get("status") or "") == "ok"
        smoothing_mae = _safe_float(smoothing.get("mae"), default=1e9)

        use_smoothing = False
        if smoothing_ok and seasonality_detected:
            use_smoothing = smoothing_mae <= (ols_mae * 1.20)
        elif smoothing_ok and float(reg["r_squared"]) < 0.30:
            use_smoothing = smoothing_mae < ols_mae

        selected_model = "ols_regression"
        slope = float(reg["slope"])
        intercept: float | None = float(reg["intercept"])
        r_squared = float(reg["r_squared"])
        residual_std = float(reg["residual_std"])
        forecast_values = ols_forecast
        deviation_pct = ols_deviation
        residuals = ols_residuals
        selected_predicted_last = predicted_last
        method_warnings: list[str] = []

        if use_smoothing and smoothing_ok:
            selected_model = str(smoothing.get("model") or "holt_linear")
            forecast_raw = smoothing.get("forecast")
            forecast_values = (
                dict(forecast_raw)
                if isinstance(forecast_raw, dict)
                else {"month_1": None, "month_2": None, "month_3": None}
            )
            slope = _safe_float(smoothing.get("slope"), default=0.0)
            intercept = None
            residuals = [
                _safe_float(item, default=0.0)
                for item in (smoothing.get("residuals") or [])
            ]
            if len(residuals) >= 2:
                residual_std = pstdev(residuals)
            else:
                residual_std = 0.0
            fitted = smoothing.get("fitted_values") or []
            fitted_tail = _safe_float(fitted[-1], default=values[-1]) if fitted else values[-1]
            selected_predicted_last = fitted_tail
            deviation_pct = (
                (values[-1] - fitted_tail) / max(abs(fitted_tail), _ZERO_GUARD)
            )

            y_mean = mean(values)
            ss_tot = sum((v - y_mean) ** 2 for v in values)
            ss_res = sum((err * err) for err in residuals)
            if ss_tot > _ZERO_GUARD:
                r_squared = max(0.0, min(1.0, 1.0 - (ss_res / ss_tot)))
            else:
                r_squared = 0.0

            if selected_model == "holt_winters_additive":
                method_warnings.append(
                    f"Selected seasonal model with period={season_length}."
                )
            else:
                method_warnings.append(
                    "Selected exponential smoothing due to weak linear fit."
                )

        confidence = _compute_confidence(
            n=n,
            r_squared=r_squared,
            cov=cov,
            tier=tier,
        )
        trend = _trend_slope_assessment(slope, avg, r_squared)
        residual_diag = residual_diagnostics(residuals, max_lag=min(6, max(2, n // 3)))

        warnings: list[str] = []
        if r_squared < 0.3:
            warnings.append(
                f"Low R^2 ({r_squared:.3f}): trend interpretation may be weak."
            )
        if str(volatility.get("regime") or "") == "high":
            warnings.append("High recent volatility detected; reliability reduced.")
        if tier == "standard":
            warnings.append(
                f"Standard tier ({n} points); confidence capped at "
                f"{_CONFIDENCE_CAP_STANDARD}. Need {MIN_POINTS_ROBUST}+ for full cap."
            )
        warnings.extend(method_warnings)
        warnings.extend(str(item) for item in residual_diag.get("warnings", []))

        return {
            "tier": tier,
            "status": "ok",
            "model": selected_model,
            "confidence_score": confidence,
            "input_points": n,
            "minimum_required": MIN_POINTS_INSUFFICIENT,
            "data_quality": {
                "coefficient_of_variation": round(cov, 6),
                "volatility": volatility,
                "rolling_average": rolling,
                "seasonality": {
                    "detected": seasonality_detected,
                    "primary_period": seasonality_period,
                    "primary_strength": seasonality.get("primary_strength"),
                },
            },
            "regression": {
                "slope": round(slope, 6),
                "intercept": round(intercept, 6) if intercept is not None else None,
                "r_squared": round(r_squared, 6),
                "residual_std": round(residual_std, 6),
            },
            "trend": trend,
            "forecast": {
                "month_1": forecast_values.get("month_1"),
                "month_2": forecast_values.get("month_2"),
                "month_3": forecast_values.get("month_3"),
            },
            "deviation_percentage": round(deviation_pct, 6),
            "slope": round(slope, 6),
            "diagnostics": {
                "method": selected_model,
                "average_value": round(avg, 6),
                "last_value": round(values[-1], 6),
                "predicted_last": round(selected_predicted_last, 6),
                "model_selection": {
                    "selected_model": selected_model,
                    "seasonality_detected": seasonality_detected,
                    "seasonality_period": seasonality_period,
                    "candidate_mae": {
                        "ols_regression": round(ols_mae, 6),
                        "smoothing": round(smoothing_mae, 6) if smoothing_ok else None,
                    },
                    "candidate_r_squared": {
                        "ols_regression": round(float(reg["r_squared"]), 6),
                        "selected": round(r_squared, 6),
                    },
                },
                "residual_diagnostics": residual_diag,
                "seasonality": seasonality,
            },
            "warnings": warnings,
        }
