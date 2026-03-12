"""
app/services/statistics/causal_inference.py

Pairwise Granger causality testing between KPI metric pairs.

Uses lagged OLS regression to test whether past values of X improve
predictions of Y beyond Y's own history.  Returns a directed causal
graph with lag structure and significance indicators.

Method
------
For each ordered pair (X → Y) at each candidate lag p:

    Restricted:   Y_t = a₀ + Σ(a_i · Y_{t-i})  for i=1..p
    Unrestricted: Y_t = a₀ + Σ(a_i · Y_{t-i}) + Σ(b_i · X_{t-i})  for i=1..p

    F = ((RSS_r - RSS_u) / p) / (RSS_u / (n - 2p - 1))

Significance is determined by comparing the F-statistic against a
critical threshold derived from the Wilson–Hilferty normal approximation
to the F-distribution CDF.

All math uses only the Python standard library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Any, Mapping, Sequence

from app.services.statistics.normalization import coerce_numeric_series

_ZERO_GUARD = 1e-9


@dataclass(frozen=True)
class GrangerConfig:
    """Configuration for Granger causality testing."""

    max_lag: int = 4
    min_observations: int = 12
    significance_alpha: float = 0.05
    max_metric_pairs: int = 20


def granger_causality(
    metric_series: Mapping[str, Sequence[Any]],
    *,
    config: GrangerConfig | None = None,
) -> dict[str, Any]:
    """
    Test pairwise Granger causality between all metric pairs.

    Returns
    -------
    dict
        status, causal_edges, summary, warnings, config
    """
    cfg = config or GrangerConfig()
    warnings: list[str] = []

    selected = _select_metrics(metric_series, min_points=cfg.min_observations)
    if len(selected) < 2:
        return {
            "status": "insufficient_data",
            "causal_edges": [],
            "summary": {
                "metrics_tested": 0,
                "pairs_tested": 0,
                "significant_edges": 0,
            },
            "warnings": ["Need at least 2 metrics with sufficient history."],
            "config": _config_dict(cfg),
        }

    # Limit pair count
    if len(selected) > cfg.max_metric_pairs:
        warnings.append(
            f"Capped metric count from {len(selected)} to "
            f"{cfg.max_metric_pairs} for tractability."
        )
        selected = selected[: cfg.max_metric_pairs]

    causal_edges: list[dict[str, Any]] = []
    pairs_tested = 0

    for cause_name in selected:
        for effect_name in selected:
            if cause_name == effect_name:
                continue

            cause_vals = coerce_numeric_series(metric_series[cause_name])
            effect_vals = coerce_numeric_series(metric_series[effect_name])
            n = min(len(cause_vals), len(effect_vals))
            if n < cfg.min_observations:
                continue

            # Align to same length (tail-aligned)
            x = cause_vals[-n:]
            y = effect_vals[-n:]
            pairs_tested += 1

            best_result = _test_granger_at_best_lag(
                x, y,
                max_lag=cfg.max_lag,
                alpha=cfg.significance_alpha,
            )
            if best_result is not None:
                causal_edges.append({
                    "cause": cause_name,
                    "effect": effect_name,
                    "lag": best_result["lag"],
                    "f_statistic": best_result["f_statistic"],
                    "p_value_approx": best_result["p_value"],
                    "significant": best_result["significant"],
                    "r_squared_improvement": best_result["r2_improvement"],
                    "direction": _direction_label(
                        best_result["coefficient_sign"],
                    ),
                    "sample_size": best_result["sample_size"],
                })

    significant_count = sum(1 for e in causal_edges if e["significant"])

    return {
        "status": "success" if pairs_tested > 0 else "insufficient_data",
        "causal_edges": sorted(
            causal_edges,
            key=lambda e: (-int(e["significant"]), -e["f_statistic"]),
        ),
        "summary": {
            "metrics_tested": len(selected),
            "pairs_tested": pairs_tested,
            "significant_edges": significant_count,
        },
        "warnings": warnings,
        "config": _config_dict(cfg),
    }


def _test_granger_at_best_lag(
    x: list[float],
    y: list[float],
    *,
    max_lag: int,
    alpha: float,
) -> dict[str, Any] | None:
    """Test Granger causality X → Y at lags 1..max_lag, return best."""
    best: dict[str, Any] | None = None

    for lag in range(1, max_lag + 1):
        n = len(y)
        if n <= 2 * lag + 1:
            continue

        # Build lagged matrices
        y_target = y[lag:]
        n_obs = len(y_target)
        if n_obs < 2 * lag + 2:
            continue

        # Restricted model: Y_t ~ Y_{t-1} ... Y_{t-lag}
        restricted_features = _build_lag_matrix(y, lag, n_obs)
        rss_r, r2_r = _ols_rss(restricted_features, y_target)

        # Unrestricted model: Y_t ~ Y_{t-1}...Y_{t-lag}, X_{t-1}...X_{t-lag}
        x_aligned = x[:len(y)]  # ensure same length
        if len(x_aligned) <= lag:
            continue
        x_lags = _build_lag_matrix(x_aligned, lag, n_obs)
        unrestricted_features = [
            r + x for r, x in zip(restricted_features, x_lags)
        ]
        rss_u, r2_u = _ols_rss(unrestricted_features, y_target)

        if rss_u < _ZERO_GUARD:
            continue

        df1 = lag
        df2 = n_obs - 2 * lag - 1
        if df2 <= 0:
            continue

        f_stat = ((rss_r - rss_u) / max(df1, 1)) / (rss_u / max(df2, 1))
        p_value = _f_survival(f_stat, df1, df2)
        significant = p_value <= alpha
        r2_improvement = max(0.0, r2_u - r2_r)

        # Get coefficient sign of X_{t-1} (first lag of cause)
        coeff_sign = _get_cause_coefficient_sign(
            unrestricted_features, y_target, lag,
        )

        result = {
            "lag": lag,
            "f_statistic": round(f_stat, 6),
            "p_value": round(p_value, 6),
            "significant": significant,
            "r2_improvement": round(r2_improvement, 6),
            "coefficient_sign": coeff_sign,
            "sample_size": n_obs,
        }

        if best is None or f_stat > best["f_statistic"]:
            best = result

    return best


def _build_lag_matrix(
    series: list[float], lag: int, n_obs: int,
) -> list[list[float]]:
    """Build [n_obs × lag] matrix of lagged values."""
    start = len(series) - n_obs
    rows: list[list[float]] = []
    for t in range(n_obs):
        idx = start + t
        row = [series[idx - k - 1] for k in range(lag) if idx - k - 1 >= 0]
        # Pad with zeros if not enough history
        while len(row) < lag:
            row.append(0.0)
        rows.append(row)
    return rows


def _ols_rss(
    features: list[list[float]], target: list[float],
) -> tuple[float, float]:
    """
    Compute RSS and R² for OLS regression with intercept.

    Uses normal equations: β = (XᵀX)⁻¹ Xᵀy
    Implemented for small feature counts via Gaussian elimination.
    """
    n = len(target)
    if n == 0:
        return 0.0, 0.0

    k = len(features[0]) if features else 0

    # Add intercept column
    X = [[1.0] + row for row in features]
    p = k + 1  # number of parameters including intercept

    if n <= p:
        ss_tot = sum((y - mean(target)) ** 2 for y in target)
        return ss_tot, 0.0

    # XᵀX
    XtX = [[0.0] * p for _ in range(p)]
    Xty = [0.0] * p
    for i in range(n):
        for j in range(p):
            Xty[j] += X[i][j] * target[i]
            for m in range(j, p):
                val = X[i][j] * X[i][m]
                XtX[j][m] += val
                if j != m:
                    XtX[m][j] += val

    # Solve via Gaussian elimination with partial pivoting
    beta = _solve_linear_system(XtX, Xty)
    if beta is None:
        ss_tot = sum((y - mean(target)) ** 2 for y in target)
        return ss_tot, 0.0

    # Compute RSS
    rss = 0.0
    y_mean = mean(target)
    ss_tot = 0.0
    for i in range(n):
        predicted = sum(beta[j] * X[i][j] for j in range(p))
        rss += (target[i] - predicted) ** 2
        ss_tot += (target[i] - y_mean) ** 2

    r2 = max(0.0, 1.0 - rss / max(ss_tot, _ZERO_GUARD))
    return rss, r2


def _solve_linear_system(
    A: list[list[float]], b: list[float],
) -> list[float] | None:
    """Solve Ax = b via Gaussian elimination with partial pivoting."""
    n = len(b)
    # Augmented matrix
    aug = [row[:] + [b[i]] for i, row in enumerate(A)]

    for col in range(n):
        # Partial pivot
        max_row = col
        max_val = abs(aug[col][col])
        for row in range(col + 1, n):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                max_row = row
        if max_val < _ZERO_GUARD:
            return None
        aug[col], aug[max_row] = aug[max_row], aug[col]

        # Eliminate
        pivot = aug[col][col]
        for row in range(col + 1, n):
            factor = aug[row][col] / pivot
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = aug[i][n]
        for j in range(i + 1, n):
            s -= aug[i][j] * x[j]
        if abs(aug[i][i]) < _ZERO_GUARD:
            return None
        x[i] = s / aug[i][i]

    return x


def _get_cause_coefficient_sign(
    features: list[list[float]],
    target: list[float],
    lag: int,
) -> int:
    """Get the sign of the first cause lag coefficient."""
    n = len(target)
    k = len(features[0]) if features else 0
    X = [[1.0] + row for row in features]
    p = k + 1

    if n <= p:
        return 0

    XtX = [[0.0] * p for _ in range(p)]
    Xty = [0.0] * p
    for i in range(n):
        for j in range(p):
            Xty[j] += X[i][j] * target[i]
            for m in range(j, p):
                val = X[i][j] * X[i][m]
                XtX[j][m] += val
                if j != m:
                    XtX[m][j] += val

    beta = _solve_linear_system(XtX, Xty)
    if beta is None:
        return 0

    # Cause coefficients start at index (1 + lag) — after intercept + Y lags
    cause_idx = 1 + lag
    if cause_idx >= len(beta):
        return 0

    coeff = beta[cause_idx]
    if coeff > _ZERO_GUARD:
        return 1
    elif coeff < -_ZERO_GUARD:
        return -1
    return 0


def _f_survival(f: float, df1: int, df2: int) -> float:
    """
    Approximate P(F > f | df1, df2) using Wilson–Hilferty transformation.

    For large df2 this is accurate; for small df it's a reasonable
    approximation sufficient for screening-level significance.
    """
    if f <= 0.0:
        return 1.0
    if df1 <= 0 or df2 <= 0:
        return 1.0

    # Transform F to approximate normal via Wilson–Hilferty
    # z ≈ ((1 - 2/(9·df2)) · (f·df1/df2)^(1/3) - (1 - 2/(9·df1)))
    #     / sqrt(2/(9·df1) + 2·(f·df1/df2)^(2/3) / (9·df2))
    a = 2.0 / (9.0 * df1)
    b = 2.0 / (9.0 * df2)
    ratio = f * df1 / df2

    try:
        ratio_third = ratio ** (1.0 / 3.0)
    except (ValueError, OverflowError):
        return 0.0

    numerator = (1.0 - b) * ratio_third - (1.0 - a)
    denominator_sq = a + b * ratio_third * ratio_third
    if denominator_sq <= 0:
        return 0.0 if f > 1.0 else 1.0

    z = numerator / math.sqrt(denominator_sq)
    return 1.0 - _norm_cdf(z)


def _norm_cdf(value: float) -> float:
    """Standard normal CDF via error function."""
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _direction_label(sign: int) -> str:
    if sign > 0:
        return "positive"
    elif sign < 0:
        return "negative"
    return "neutral"


def _select_metrics(
    metric_series: Mapping[str, Sequence[Any]],
    *,
    min_points: int,
) -> list[str]:
    """Select metrics with sufficient data, sorted by length desc."""
    candidates: list[tuple[str, int]] = []
    for name, values in metric_series.items():
        series = coerce_numeric_series(values)
        if len(series) >= min_points:
            candidates.append((name, len(series)))
    candidates.sort(key=lambda t: (-t[1], t[0]))
    return [name for name, _ in candidates]


def _config_dict(cfg: GrangerConfig) -> dict[str, Any]:
    return {
        "max_lag": cfg.max_lag,
        "min_observations": cfg.min_observations,
        "significance_alpha": cfg.significance_alpha,
        "max_metric_pairs": cfg.max_metric_pairs,
    }
