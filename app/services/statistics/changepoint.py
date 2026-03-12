"""
app/services/statistics/changepoint.py

Statistical changepoint detection for time-series.

Implements two complementary methods:

    1. **CUSUM (Cumulative Sum)** — detects sustained level shifts by
       tracking cumulative deviations from the target mean.

    2. **Binary segmentation** — recursive optimal split-point detection
       that minimises within-segment variance (PELT-inspired but O(n²)
       for simplicity and stdlib-only constraint).

Both methods return changepoint locations with significance measures.

All math uses only the Python standard library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any, Sequence

from app.services.statistics.normalization import coerce_numeric_series

_ZERO_GUARD = 1e-9


@dataclass(frozen=True)
class ChangepointConfig:
    """Configuration for changepoint detection."""

    min_segment_size: int = 4
    max_changepoints: int = 5
    cusum_threshold: float = 4.0  # standard deviations
    binseg_penalty_factor: float = 2.0  # BIC-like penalty multiplier
    min_improvement_ratio: float = 0.15  # min variance reduction to accept


def detect_changepoints(
    values: Sequence[Any],
    *,
    config: ChangepointConfig | None = None,
) -> dict[str, Any]:
    """
    Detect changepoints using both CUSUM and binary segmentation.

    Parameters
    ----------
    values:
        Ordered numeric observations, oldest first.
    config:
        Optional configuration overrides.

    Returns
    -------
    dict
        changepoints, cusum_result, binseg_result, combined,
        summary, warnings
    """
    cfg = config or ChangepointConfig()
    series = coerce_numeric_series(values)
    n = len(series)
    warnings: list[str] = []

    if n < cfg.min_segment_size * 2:
        return {
            "status": "insufficient_data",
            "changepoints": [],
            "cusum_result": {"detected": False, "changepoints": []},
            "binseg_result": {"detected": False, "changepoints": []},
            "combined": [],
            "summary": {
                "series_length": n,
                "changepoints_detected": 0,
                "methods_agreeing": 0,
            },
            "warnings": [
                f"Need at least {cfg.min_segment_size * 2} points; got {n}."
            ],
        }

    # Method 1: CUSUM
    cusum_result = _cusum_detect(series, cfg)

    # Method 2: Binary segmentation
    binseg_result = _binary_segmentation(series, cfg)

    # Combine: union of detected changepoints, merge nearby
    cusum_cps = set(cusum_result.get("changepoint_indices", []))
    binseg_cps = set(binseg_result.get("changepoint_indices", []))
    all_cps = cusum_cps | binseg_cps
    merged = _merge_nearby(sorted(all_cps), tolerance=max(2, cfg.min_segment_size // 2))

    # Classify each changepoint
    combined: list[dict[str, Any]] = []
    for cp in merged:
        methods = []
        if cp in cusum_cps or any(abs(cp - c) <= 2 for c in cusum_cps):
            methods.append("cusum")
        if cp in binseg_cps or any(abs(cp - c) <= 2 for c in binseg_cps):
            methods.append("binary_segmentation")

        left = series[max(0, cp - cfg.min_segment_size):cp]
        right = series[cp:min(n, cp + cfg.min_segment_size)]
        shift = _level_shift(left, right)

        combined.append({
            "index": cp,
            "methods": methods,
            "methods_agreeing": len(methods),
            "confirmed": len(methods) >= 2,
            "level_shift": shift,
        })

    methods_agreeing = sum(1 for c in combined if c["confirmed"])

    return {
        "status": "success",
        "changepoints": [c["index"] for c in combined],
        "cusum_result": cusum_result,
        "binseg_result": binseg_result,
        "combined": combined,
        "summary": {
            "series_length": n,
            "changepoints_detected": len(combined),
            "methods_agreeing": methods_agreeing,
        },
        "warnings": warnings,
    }


def _cusum_detect(series: list[float], cfg: ChangepointConfig) -> dict[str, Any]:
    """CUSUM changepoint detection."""
    n = len(series)
    mu = mean(series)
    sigma = pstdev(series) if n >= 2 else 1.0
    if sigma < _ZERO_GUARD:
        return {
            "detected": False,
            "changepoints": [],
            "changepoint_indices": [],
            "cusum_values": [],
        }

    threshold = cfg.cusum_threshold * sigma

    # Compute CUSUM
    cusum_pos = [0.0] * n
    cusum_neg = [0.0] * n
    changepoint_indices: list[int] = []

    for i in range(1, n):
        cusum_pos[i] = max(0.0, cusum_pos[i - 1] + (series[i] - mu))
        cusum_neg[i] = min(0.0, cusum_neg[i - 1] + (series[i] - mu))

        if cusum_pos[i] > threshold or abs(cusum_neg[i]) > threshold:
            changepoint_indices.append(i)
            # Reset after detection
            cusum_pos[i] = 0.0
            cusum_neg[i] = 0.0

    # Filter too-close changepoints
    filtered = _merge_nearby(changepoint_indices, tolerance=cfg.min_segment_size)

    cusum_values = [
        round(max(cusum_pos[i], abs(cusum_neg[i])), 6)
        for i in range(n)
    ]

    return {
        "detected": len(filtered) > 0,
        "changepoints": [
            {
                "index": idx,
                "cusum_magnitude": cusum_values[idx] if idx < n else 0.0,
                "type": "level_shift",
            }
            for idx in filtered[:cfg.max_changepoints]
        ],
        "changepoint_indices": filtered[:cfg.max_changepoints],
        "cusum_values": cusum_values,
    }


def _binary_segmentation(
    series: list[float], cfg: ChangepointConfig,
) -> dict[str, Any]:
    """Binary segmentation for optimal split-point detection."""
    changepoints: list[int] = []
    _binseg_recurse(
        series,
        start=0,
        end=len(series),
        changepoints=changepoints,
        min_segment=cfg.min_segment_size,
        max_changepoints=cfg.max_changepoints,
        min_improvement=cfg.min_improvement_ratio,
        penalty_factor=cfg.binseg_penalty_factor,
    )
    changepoints.sort()

    return {
        "detected": len(changepoints) > 0,
        "changepoints": [
            {"index": cp, "type": "variance_reduction"}
            for cp in changepoints
        ],
        "changepoint_indices": changepoints,
    }


def _binseg_recurse(
    series: list[float],
    *,
    start: int,
    end: int,
    changepoints: list[int],
    min_segment: int,
    max_changepoints: int,
    min_improvement: float,
    penalty_factor: float,
) -> None:
    """Recursively find optimal split point."""
    if len(changepoints) >= max_changepoints:
        return
    if end - start < min_segment * 2:
        return

    segment = series[start:end]
    n = len(segment)
    total_var = _segment_cost(segment)
    if total_var < _ZERO_GUARD:
        return

    best_split = -1
    best_reduction = 0.0
    penalty = penalty_factor * math.log(max(n, 2))

    for split in range(min_segment, n - min_segment + 1):
        left = segment[:split]
        right = segment[split:]
        cost = _segment_cost(left) + _segment_cost(right) + penalty
        reduction = (total_var - cost) / max(total_var, _ZERO_GUARD)

        if reduction > best_reduction:
            best_reduction = reduction
            best_split = split

    if best_split >= 0 and best_reduction >= min_improvement:
        cp = start + best_split
        changepoints.append(cp)

        # Recurse on both segments
        _binseg_recurse(
            series,
            start=start,
            end=cp,
            changepoints=changepoints,
            min_segment=min_segment,
            max_changepoints=max_changepoints,
            min_improvement=min_improvement,
            penalty_factor=penalty_factor,
        )
        _binseg_recurse(
            series,
            start=cp,
            end=end,
            changepoints=changepoints,
            min_segment=min_segment,
            max_changepoints=max_changepoints,
            min_improvement=min_improvement,
            penalty_factor=penalty_factor,
        )


def _segment_cost(values: list[float]) -> float:
    """Sum of squared deviations from segment mean."""
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return sum((v - mu) ** 2 for v in values)


def _level_shift(left: list[float], right: list[float]) -> dict[str, Any]:
    """Compute level shift between two segments."""
    if not left or not right:
        return {"magnitude": 0.0, "direction": "none", "relative_shift": 0.0}

    left_mean = mean(left)
    right_mean = mean(right)
    magnitude = right_mean - left_mean
    baseline = max(abs(left_mean), _ZERO_GUARD)
    relative = magnitude / baseline

    if relative > 0.02:
        direction = "increase"
    elif relative < -0.02:
        direction = "decrease"
    else:
        direction = "stable"

    return {
        "magnitude": round(magnitude, 6),
        "direction": direction,
        "relative_shift": round(relative, 6),
        "left_mean": round(left_mean, 6),
        "right_mean": round(right_mean, 6),
    }


def _merge_nearby(indices: list[int], tolerance: int) -> list[int]:
    """Merge changepoints that are within tolerance of each other."""
    if not indices:
        return []
    merged = [indices[0]]
    for idx in indices[1:]:
        if idx - merged[-1] >= tolerance:
            merged.append(idx)
    return merged
