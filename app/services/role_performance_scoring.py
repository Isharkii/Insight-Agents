"""
app/services/role_performance_scoring.py

Deterministic role performance scoring using pure NumPy math.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


DEFAULT_WEIGHTS: dict[str, float] = {
    "growth_rate": 0.30,
    "efficiency_metric": 0.25,
    "stability": 0.25,
    "contribution_weight": 0.20,
}


def _as_numeric_array(value: Any) -> np.ndarray:
    """Coerce scalar/list-like input to a finite float NumPy array."""
    if value is None:
        return np.array([], dtype=float)
    if isinstance(value, np.ndarray):
        arr = value.astype(float, copy=False)
    elif isinstance(value, (list, tuple)):
        if not value:
            return np.array([], dtype=float)
        try:
            arr = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return np.array([], dtype=float)
    else:
        try:
            arr = np.asarray([float(value)], dtype=float)
        except (TypeError, ValueError):
            return np.array([], dtype=float)

    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr[np.isfinite(arr)]


def _safe_mean(arr: np.ndarray, default: float = 0.0) -> float:
    if arr.size == 0:
        return default
    return float(np.mean(arr))


def _safe_variance(arr: np.ndarray, default: float = 0.0) -> float:
    if arr.size <= 1:
        return default
    return float(np.var(arr))


def _minmax_0_100(values: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 100]. Returns neutral 50s for flat vectors."""
    if values.size == 0:
        return np.array([], dtype=float)
    min_v = float(np.min(values))
    max_v = float(np.max(values))
    if np.isclose(max_v, min_v):
        return np.full(values.shape, 50.0, dtype=float)
    return ((values - min_v) / (max_v - min_v)) * 100.0


def _resolve_weights(
    category: str | None,
    category_weight_overrides: Mapping[str, Mapping[str, float]] | None,
) -> dict[str, float]:
    weights = dict(DEFAULT_WEIGHTS)
    if category and category_weight_overrides:
        override = category_weight_overrides.get(str(category).strip().lower())
        if isinstance(override, Mapping):
            for key in DEFAULT_WEIGHTS:
                if key in override:
                    try:
                        weights[key] = float(override[key])
                    except (TypeError, ValueError):
                        continue

    # Clamp negatives and re-normalize to sum to 1.0.
    for key, value in tuple(weights.items()):
        weights[key] = max(0.0, float(value))
    total = float(np.sum(np.fromiter(weights.values(), dtype=float)))
    if total <= 0.0:
        return dict(DEFAULT_WEIGHTS)
    for key in weights:
        weights[key] = weights[key] / total
    return weights


def _classify(score: float) -> str:
    if score >= 80.0:
        return "strong"
    if score >= 60.0:
        return "stable"
    if score >= 40.0:
        return "at_risk"
    return "critical"


class RolePerformanceScorer:
    """
    Compute per-role performance score (0-100) from weighted components:
      - growth_rate         (30%)
      - efficiency_metric   (25%)
      - stability           (25%, lower variance => higher score)
      - contribution_weight (20%)
    """

    def score(
        self,
        role_metrics: Mapping[str, Mapping[str, Any]],
        *,
        category: str | None = None,
        category_weight_overrides: Mapping[str, Mapping[str, float]] | None = None,
    ) -> dict[str, dict[str, float | str]]:
        if not role_metrics:
            return {}

        roles = list(role_metrics.keys())
        n_roles = len(roles)

        growth_raw = np.zeros(n_roles, dtype=float)
        efficiency_raw = np.zeros(n_roles, dtype=float)
        contribution_raw = np.zeros(n_roles, dtype=float)
        variance_raw = np.zeros(n_roles, dtype=float)

        for idx, role in enumerate(roles):
            metrics = role_metrics.get(role) or {}

            growth_arr = _as_numeric_array(metrics.get("growth_rate"))
            efficiency_arr = _as_numeric_array(metrics.get("efficiency_metric"))
            contribution_arr = _as_numeric_array(metrics.get("contribution_weight"))

            # Prefer explicit stability series for variance; otherwise fall back.
            stability_series = _as_numeric_array(metrics.get("stability_series"))
            if stability_series.size == 0:
                stability_series = (
                    growth_arr if growth_arr.size > 1 else efficiency_arr
                )

            growth_raw[idx] = _safe_mean(growth_arr, default=0.0)
            efficiency_raw[idx] = _safe_mean(efficiency_arr, default=0.0)
            contribution_raw[idx] = _safe_mean(contribution_arr, default=0.0)
            variance_raw[idx] = _safe_variance(stability_series, default=0.0)

        growth_score = _minmax_0_100(growth_raw)
        efficiency_score = _minmax_0_100(efficiency_raw)
        contribution_score = _minmax_0_100(contribution_raw)

        # Low variance should be high stability score: invert before scaling.
        stability_score = _minmax_0_100(-variance_raw)

        weights = _resolve_weights(category, category_weight_overrides)
        performance_index = (
            growth_score * weights["growth_rate"]
            + efficiency_score * weights["efficiency_metric"]
            + stability_score * weights["stability"]
            + contribution_score * weights["contribution_weight"]
        )

        output: dict[str, dict[str, float | str]] = {}
        for idx, role in enumerate(roles):
            score = float(np.clip(performance_index[idx], 0.0, 100.0))
            output[role] = {
                "performance_score": round(score, 2),
                "classification": _classify(score),
            }
        return output


def score_role_performance(
    role_metrics: Mapping[str, Mapping[str, Any]],
    *,
    category: str | None = None,
    category_weight_overrides: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, dict[str, float | str]]:
    """Convenience function wrapper around RolePerformanceScorer."""
    return RolePerformanceScorer().score(
        role_metrics,
        category=category,
        category_weight_overrides=category_weight_overrides,
    )

