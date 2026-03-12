"""
forecast/diagnostics.py

Residual diagnostics for deterministic forecast models.
"""

from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Any, Sequence

_ZERO_GUARD = 1e-9


def durbin_watson_statistic(residuals: Sequence[float]) -> float | None:
    """
    Compute Durbin-Watson statistic for first-order autocorrelation.

    DW ~ 2: no autocorrelation, <2: positive autocorrelation, >2: negative.
    """
    values = [float(v) for v in residuals if _is_finite(v)]
    if len(values) < 2:
        return None

    numerator = 0.0
    denominator = 0.0
    for prev, curr in zip(values[:-1], values[1:]):
        numerator += (curr - prev) ** 2
    for value in values:
        denominator += value * value
    if denominator < _ZERO_GUARD:
        return None
    return round(numerator / denominator, 6)


def ljung_box_test(
    residuals: Sequence[float],
    *,
    max_lag: int = 6,
) -> dict[str, Any]:
    """
    Ljung-Box whiteness test with a chi-square p-value approximation.
    """
    values = [float(v) for v in residuals if _is_finite(v)]
    n = len(values)
    if n < 8:
        return {
            "status": "insufficient_data",
            "lags_tested": 0,
            "q_statistic": None,
            "p_value_approx": None,
            "autocorrelation_detected": False,
        }

    lag = min(max_lag, max(1, n // 3))
    acf_values: list[float] = []
    for k in range(1, lag + 1):
        acf_values.append(_autocorrelation(values, lag=k))

    q_stat = 0.0
    for k, rk in enumerate(acf_values, start=1):
        q_stat += (rk * rk) / max(n - k, 1)
    q_stat *= n * (n + 2)

    dof = max(1, lag)
    p_value = _chi_square_survival(q_stat, dof)
    autocorrelation_detected = bool(p_value < 0.05)

    return {
        "status": "ok",
        "lags_tested": lag,
        "q_statistic": round(q_stat, 6),
        "p_value_approx": round(p_value, 6),
        "autocorrelation_detected": autocorrelation_detected,
        "acf": [round(v, 6) for v in acf_values],
    }


def residual_diagnostics(
    residuals: Sequence[float],
    *,
    max_lag: int = 6,
) -> dict[str, Any]:
    """
    Unified residual diagnostics payload for forecast models.
    """
    values = [float(v) for v in residuals if _is_finite(v)]
    if len(values) < 3:
        return {
            "status": "insufficient_data",
            "residual_count": len(values),
            "residual_mean": round(mean(values), 6) if values else None,
            "residual_std": round(pstdev(values), 6) if len(values) >= 2 else None,
            "durbin_watson": None,
            "durbin_watson_label": "insufficient_data",
            "ljung_box": {
                "status": "insufficient_data",
                "lags_tested": 0,
                "q_statistic": None,
                "p_value_approx": None,
                "autocorrelation_detected": False,
            },
            "warnings": ["Need at least 3 residuals for diagnostics."],
        }

    dw = durbin_watson_statistic(values)
    ljung = ljung_box_test(values, max_lag=max_lag)

    warnings: list[str] = []
    if dw is not None and (dw < 1.2 or dw > 2.8):
        warnings.append(
            f"Durbin-Watson indicates possible autocorrelation (dw={dw:.3f})."
        )
    if bool(ljung.get("autocorrelation_detected")):
        warnings.append(
            "Ljung-Box test indicates residual autocorrelation."
        )

    return {
        "status": "ok",
        "residual_count": len(values),
        "residual_mean": round(mean(values), 6),
        "residual_std": round(pstdev(values), 6) if len(values) >= 2 else 0.0,
        "durbin_watson": dw,
        "durbin_watson_label": _durbin_watson_label(dw),
        "ljung_box": ljung,
        "warnings": warnings,
    }


def _durbin_watson_label(value: float | None) -> str:
    if value is None:
        return "insufficient_data"
    if value < 1.2:
        return "positive_autocorrelation"
    if value > 2.8:
        return "negative_autocorrelation"
    return "no_strong_autocorrelation"


def _autocorrelation(values: Sequence[float], *, lag: int) -> float:
    n = len(values)
    if lag <= 0 or lag >= n:
        return 0.0
    mu = mean(values)
    numerator = 0.0
    denominator = 0.0
    for idx in range(lag, n):
        numerator += (values[idx] - mu) * (values[idx - lag] - mu)
    for val in values:
        denominator += (val - mu) ** 2
    if denominator < _ZERO_GUARD:
        return 0.0
    return numerator / denominator


def _chi_square_survival(x: float, dof: int) -> float:
    """
    Chi-square survival function approximation via Wilson-Hilferty transform.
    """
    if x <= 0.0:
        return 1.0
    k = max(1, int(dof))
    z = (
        ((x / k) ** (1.0 / 3.0))
        - (1.0 - (2.0 / (9.0 * k)))
    ) / math.sqrt(2.0 / (9.0 * k))
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return max(0.0, min(1.0, 1.0 - cdf))


def _is_finite(value: object) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric)

