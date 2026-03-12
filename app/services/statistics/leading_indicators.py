"""
app/services/statistics/leading_indicators.py

Cross-correlation analysis for leading vs lagging indicator detection.

For each metric pair, computes the cross-correlation function (CCF) at
multiple lags to identify temporal relationships.  Metrics are classified
as leading, coincident, or lagging relative to each other.

All math uses only the Python standard library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence

from app.services.statistics.normalization import coerce_numeric_series

_ZERO_GUARD = 1e-9


@dataclass(frozen=True)
class LeadingIndicatorConfig:
    """Configuration for leading indicator detection."""

    max_lag: int = 6
    min_observations: int = 8
    significance_z: float = 1.96  # 95% confidence
    min_abs_correlation: float = 0.30
    max_metrics: int = 10


def detect_leading_indicators(
    metric_series: Mapping[str, Sequence[Any]],
    *,
    config: LeadingIndicatorConfig | None = None,
) -> dict[str, Any]:
    """
    Detect leading and lagging relationships between metric pairs.

    Returns
    -------
    dict
        relationships, metric_classifications, summary, warnings
    """
    cfg = config or LeadingIndicatorConfig()
    warnings: list[str] = []

    selected = _select_metrics(metric_series, min_points=cfg.min_observations, max_count=cfg.max_metrics)
    if len(selected) < 2:
        return {
            "status": "insufficient_data",
            "relationships": [],
            "metric_classifications": {},
            "summary": {"metrics_analyzed": 0, "significant_relationships": 0},
            "warnings": ["Need at least 2 metrics with sufficient history."],
        }

    relationships: list[dict[str, Any]] = []

    for i, name_a in enumerate(selected):
        for j, name_b in enumerate(selected):
            if j <= i:
                continue

            a = coerce_numeric_series(metric_series[name_a])
            b = coerce_numeric_series(metric_series[name_b])
            n = min(len(a), len(b))
            if n < cfg.min_observations:
                continue

            x = a[-n:]
            y = b[-n:]

            ccf_result = _cross_correlation_function(
                x, y,
                max_lag=cfg.max_lag,
                significance_z=cfg.significance_z,
                min_abs_correlation=cfg.min_abs_correlation,
            )

            if ccf_result["optimal_lag"] is not None:
                relationships.append({
                    "metric_a": name_a,
                    "metric_b": name_b,
                    "optimal_lag": ccf_result["optimal_lag"],
                    "optimal_correlation": ccf_result["optimal_correlation"],
                    "relationship": ccf_result["relationship"],
                    "significant": ccf_result["significant"],
                    "ccf_values": ccf_result["ccf_values"],
                    "significance_threshold": ccf_result["significance_threshold"],
                    "sample_size": n,
                })

    # Classify each metric as leading, coincident, or lagging
    classifications = _classify_metrics(relationships, selected)
    significant_count = sum(1 for r in relationships if r["significant"])

    return {
        "status": "success" if relationships else "insufficient_data",
        "relationships": sorted(
            relationships,
            key=lambda r: (-int(r["significant"]), -abs(r["optimal_correlation"])),
        ),
        "metric_classifications": classifications,
        "summary": {
            "metrics_analyzed": len(selected),
            "significant_relationships": significant_count,
            "total_relationships": len(relationships),
        },
        "warnings": warnings,
    }


def _cross_correlation_function(
    x: list[float],
    y: list[float],
    *,
    max_lag: int,
    significance_z: float,
    min_abs_correlation: float,
) -> dict[str, Any]:
    """
    Compute CCF between x and y at lags -max_lag..+max_lag.

    Positive lag k means x leads y by k periods.
    Negative lag k means y leads x by |k| periods.
    """
    n = len(x)
    if n < 3:
        return {
            "optimal_lag": None,
            "optimal_correlation": 0.0,
            "relationship": "insufficient_data",
            "significant": False,
            "ccf_values": {},
            "significance_threshold": 0.0,
        }

    mx = mean(x)
    my = mean(y)
    sx = pstdev(x)
    sy = pstdev(y)

    if sx < _ZERO_GUARD or sy < _ZERO_GUARD:
        return {
            "optimal_lag": 0,
            "optimal_correlation": 0.0,
            "relationship": "constant_series",
            "significant": False,
            "ccf_values": {},
            "significance_threshold": 0.0,
        }

    significance_threshold = significance_z / math.sqrt(n)

    ccf_values: dict[int, float] = {}
    best_lag = 0
    best_abs_corr = 0.0
    best_corr = 0.0

    for lag in range(-max_lag, max_lag + 1):
        # At lag k: corr(x[t], y[t+k])
        # Positive k: x leads y
        # Negative k: y leads x
        if lag >= 0:
            x_slice = x[:n - lag] if lag > 0 else x
            y_slice = y[lag:] if lag > 0 else y
        else:
            x_slice = x[-lag:]
            y_slice = y[:n + lag]

        m = len(x_slice)
        if m < 3:
            continue

        lmx = mean(x_slice)
        lmy = mean(y_slice)

        numerator = sum(
            (x_slice[i] - lmx) * (y_slice[i] - lmy) for i in range(m)
        )
        denom_x = sum((v - lmx) ** 2 for v in x_slice)
        denom_y = sum((v - lmy) ** 2 for v in y_slice)
        denom = math.sqrt(max(denom_x * denom_y, _ZERO_GUARD))

        r = numerator / denom
        r = max(-1.0, min(1.0, r))
        ccf_values[lag] = round(r, 6)

        if abs(r) > best_abs_corr:
            best_abs_corr = abs(r)
            best_corr = r
            best_lag = lag

    significant = (
        best_abs_corr >= min_abs_correlation
        and best_abs_corr >= significance_threshold
    )

    if best_lag > 0:
        relationship = "a_leads_b"
    elif best_lag < 0:
        relationship = "b_leads_a"
    else:
        relationship = "coincident"

    return {
        "optimal_lag": best_lag,
        "optimal_correlation": round(best_corr, 6),
        "relationship": relationship,
        "significant": significant,
        "ccf_values": ccf_values,
        "significance_threshold": round(significance_threshold, 6),
    }


def _classify_metrics(
    relationships: list[dict[str, Any]],
    all_metrics: list[str],
) -> dict[str, dict[str, Any]]:
    """Classify each metric as leading, coincident, or lagging."""
    lead_counts: dict[str, int] = {m: 0 for m in all_metrics}
    lag_counts: dict[str, int] = {m: 0 for m in all_metrics}
    coincident_counts: dict[str, int] = {m: 0 for m in all_metrics}

    for rel in relationships:
        if not rel.get("significant"):
            continue

        a = rel["metric_a"]
        b = rel["metric_b"]
        relationship = rel["relationship"]

        if relationship == "a_leads_b":
            lead_counts[a] = lead_counts.get(a, 0) + 1
            lag_counts[b] = lag_counts.get(b, 0) + 1
        elif relationship == "b_leads_a":
            lead_counts[b] = lead_counts.get(b, 0) + 1
            lag_counts[a] = lag_counts.get(a, 0) + 1
        elif relationship == "coincident":
            coincident_counts[a] = coincident_counts.get(a, 0) + 1
            coincident_counts[b] = coincident_counts.get(b, 0) + 1

    classifications: dict[str, dict[str, Any]] = {}
    for m in all_metrics:
        leads = lead_counts.get(m, 0)
        lags = lag_counts.get(m, 0)
        coinc = coincident_counts.get(m, 0)

        if leads > lags and leads > coinc:
            role = "leading"
        elif lags > leads and lags > coinc:
            role = "lagging"
        elif coinc > 0 and coinc >= leads and coinc >= lags:
            role = "coincident"
        else:
            role = "unclassified"

        classifications[m] = {
            "role": role,
            "leads_count": leads,
            "lags_count": lags,
            "coincident_count": coinc,
        }

    return classifications


def _select_metrics(
    metric_series: Mapping[str, Sequence[Any]],
    *,
    min_points: int,
    max_count: int,
) -> list[str]:
    """Select metrics with sufficient data, sorted by length desc."""
    candidates: list[tuple[str, int]] = []
    for name, values in metric_series.items():
        series = coerce_numeric_series(values)
        if len(series) >= min_points:
            candidates.append((name, len(series)))
    candidates.sort(key=lambda t: (-t[1], t[0]))
    return [name for name, _ in candidates[:max_count]]
