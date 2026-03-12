from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from app.services.statistics.normalization import coerce_numeric_series


_BUSINESS_RULES_PATH = Path(__file__).resolve().parents[3] / "config" / "business_rules.yaml"


@lru_cache(maxsize=1)
def _load_business_rules() -> dict[str, Any]:
    try:
        raw = _BUSINESS_RULES_PATH.read_text(encoding="utf-8")
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else {}
    except (OSError, TypeError, ValueError):
        return {}


def _as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_int(value: object, default: int, *, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, parsed)


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower()


_MULTI_RULES = _as_dict(_load_business_rules().get("multivariate"))
_MULTI_DEFAULTS = _as_dict(_MULTI_RULES.get("defaults"))


@dataclass(frozen=True)
class MultivariateConfig:
    min_shared_points: int = _as_int(_MULTI_DEFAULTS.get("min_shared_points"), 5)
    significance_alpha: float = _as_float(_MULTI_DEFAULTS.get("significance_alpha"), 0.10)
    min_abs_correlation: float = _as_float(_MULTI_DEFAULTS.get("min_abs_correlation"), 0.25)
    max_metrics: int = _as_int(_MULTI_DEFAULTS.get("max_metrics"), 8)
    zero_guard: float = _as_float(_MULTI_DEFAULTS.get("zero_guard"), 1e-9)
    correlation_method: str = str(_MULTI_DEFAULTS.get("correlation_method") or "spearman")


def compute_multivariate_context(
    metric_series: Mapping[str, Sequence[Any]],
    *,
    segment_rows: Sequence[Mapping[str, Any]] = (),
    preferred_metric_candidates: Sequence[str] = (),
    config: MultivariateConfig | None = None,
) -> dict[str, Any]:
    cfg = config or MultivariateConfig()
    selected_metrics = _select_metrics(
        metric_series,
        preferred_metric_candidates=preferred_metric_candidates,
        max_metrics=cfg.max_metrics,
    )

    if len(selected_metrics) < 2:
        return {
            "status": "partial",
            "confidence_score": 0.4,
            "warnings": ["Need at least two metrics for multivariate analysis."],
            "correlation": {
                "method": "spearman",
                "matrix": {},
                "filtered_matrix": {},
                "pearson_matrix": {},
                "spearman_matrix": {},
                "pairs": [],
                "significance_alpha": cfg.significance_alpha,
                "min_abs_correlation": cfg.min_abs_correlation,
                "insufficient_history": True,
            },
            "variance_decomposition": {
                "total_variance": 0.0,
                "metric_variances": {},
                "variance_share": {},
            },
            "segment_contribution": _segment_contribution(segment_rows),
        }

    pair_details: list[dict[str, Any]] = []
    matrix: dict[str, dict[str, float | None]] = {name: {} for name in selected_metrics}
    filtered: dict[str, dict[str, float | None]] = {name: {} for name in selected_metrics}
    pearson_matrix: dict[str, dict[str, float | None]] = {name: {} for name in selected_metrics}
    spearman_matrix: dict[str, dict[str, float | None]] = {name: {} for name in selected_metrics}
    insufficient_pairs = 0
    significant_pairs = 0
    method = "spearman" if str(cfg.correlation_method).strip().lower() != "pearson" else "pearson"

    for i, m1 in enumerate(selected_metrics):
        for j, m2 in enumerate(selected_metrics):
            if j < i:
                continue
            if m1 == m2:
                matrix[m1][m2] = 1.0
                filtered[m1][m2] = 1.0
                pearson_matrix[m1][m2] = 1.0
                spearman_matrix[m1][m2] = 1.0
                continue
            pearson_r, n = _pearson_tail(metric_series.get(m1, ()), metric_series.get(m2, ()))
            spearman_r, spearman_n = _spearman_tail(metric_series.get(m1, ()), metric_series.get(m2, ()))
            active_r = spearman_r if method == "spearman" else pearson_r
            active_n = spearman_n if method == "spearman" else n
            p_value = _p_value_from_correlation(active_r, active_n) if active_r is not None else None
            significant = (
                active_r is not None
                and active_n >= cfg.min_shared_points
                and abs(active_r) >= cfg.min_abs_correlation
                and p_value is not None
                and p_value <= cfg.significance_alpha
            )
            if active_n < cfg.min_shared_points or active_r is None:
                insufficient_pairs += 1
            if significant:
                significant_pairs += 1

            corr_value = round(active_r, 6) if active_r is not None else None
            matrix[m1][m2] = corr_value
            matrix[m2][m1] = corr_value
            filtered[m1][m2] = corr_value if significant else None
            filtered[m2][m1] = corr_value if significant else None

            pearson_value = round(pearson_r, 6) if pearson_r is not None else None
            spearman_value = round(spearman_r, 6) if spearman_r is not None else None
            pearson_matrix[m1][m2] = pearson_value
            pearson_matrix[m2][m1] = pearson_value
            spearman_matrix[m1][m2] = spearman_value
            spearman_matrix[m2][m1] = spearman_value

            pair_details.append(
                {
                    "metric_x": m1,
                    "metric_y": m2,
                    "correlation_method": method,
                    "correlation": corr_value,
                    "sample_size": active_n,
                    "p_value": round(p_value, 6) if p_value is not None else None,
                    "significant": significant,
                    "pearson_correlation": pearson_value,
                    "spearman_correlation": spearman_value,
                }
            )

    variance = _variance_decomposition(metric_series, selected_metrics)
    segment = _segment_contribution(segment_rows)
    total_pairs = max(1, len(pair_details))
    confidence = max(
        0.2,
        round(
            1.0
            - (insufficient_pairs / total_pairs) * 0.4
            - ((total_pairs - significant_pairs) / total_pairs) * 0.2,
            6,
        ),
    )
    status = "partial" if insufficient_pairs > 0 else "success"
    warnings: list[str] = []
    if insufficient_pairs > 0:
        warnings.append(
            f"{insufficient_pairs} metric pairs below minimum shared history ({cfg.min_shared_points})."
        )

    return {
        "status": status,
        "confidence_score": confidence,
        "warnings": warnings,
        "correlation": {
            "method": method,
            "matrix": matrix,
            "filtered_matrix": filtered,
            "pearson_matrix": pearson_matrix,
            "spearman_matrix": spearman_matrix,
            "pairs": pair_details,
            "significance_alpha": cfg.significance_alpha,
            "min_abs_correlation": cfg.min_abs_correlation,
            "insufficient_history": insufficient_pairs > 0,
            "significant_pairs": significant_pairs,
            "total_pairs": len(pair_details),
        },
        "variance_decomposition": variance,
        "segment_contribution": segment,
    }


def _select_metrics(
    metric_series: Mapping[str, Sequence[Any]],
    *,
    preferred_metric_candidates: Sequence[str],
    max_metrics: int,
) -> list[str]:
    available = [name for name, values in metric_series.items() if len(coerce_numeric_series(values)) >= 2]
    if not available:
        return []

    chosen: list[str] = []
    preferred = [_normalize(v) for v in preferred_metric_candidates if _normalize(v)]
    for candidate in preferred:
        for name in available:
            if _normalize(name) == candidate and name not in chosen:
                chosen.append(name)

    ranked = sorted(
        available,
        key=lambda name: (-len(coerce_numeric_series(metric_series.get(name, ()))), name),
    )
    for name in ranked:
        if name not in chosen:
            chosen.append(name)
        if len(chosen) >= max_metrics:
            break
    return chosen[:max_metrics]


def _pearson_tail(values_a: Sequence[Any], values_b: Sequence[Any]) -> tuple[float | None, int]:
    a = coerce_numeric_series(values_a)
    b = coerce_numeric_series(values_b)
    n = min(len(a), len(b))
    if n < 2:
        return None, n
    x = a[-n:]
    y = b[-n:]
    mx = mean(x)
    my = mean(y)
    numerator = 0.0
    denom_x = 0.0
    denom_y = 0.0
    for xv, yv in zip(x, y):
        dx = xv - mx
        dy = yv - my
        numerator += dx * dy
        denom_x += dx * dx
        denom_y += dy * dy
    denominator = math.sqrt(max(denom_x * denom_y, 0.0))
    if denominator == 0.0:
        return None, n
    r = numerator / denominator
    r = max(-1.0, min(1.0, r))
    return r, n


def _spearman_tail(values_a: Sequence[Any], values_b: Sequence[Any]) -> tuple[float | None, int]:
    a = coerce_numeric_series(values_a)
    b = coerce_numeric_series(values_b)
    n = min(len(a), len(b))
    if n < 2:
        return None, n
    x = a[-n:]
    y = b[-n:]
    x_rank = _rankdata(x)
    y_rank = _rankdata(y)
    return _pearson_tail(x_rank, y_rank)


def _rankdata(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0 for _ in values]
    idx = 0
    while idx < len(indexed):
        end = idx
        while end + 1 < len(indexed) and indexed[end + 1][1] == indexed[idx][1]:
            end += 1
        avg_rank = (idx + end + 2) / 2.0
        for pos in range(idx, end + 1):
            original_index = indexed[pos][0]
            ranks[original_index] = avg_rank
        idx = end + 1
    return ranks


def _p_value_from_correlation(r: float, n: int) -> float | None:
    if n <= 3:
        return None
    if abs(r) >= 1.0:
        return 0.0
    z = 0.5 * math.log((1.0 + r) / (1.0 - r)) * math.sqrt(max(n - 3, 1))
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return max(0.0, min(1.0, p))


def _norm_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _variance_decomposition(
    metric_series: Mapping[str, Sequence[Any]],
    metrics: Sequence[str],
) -> dict[str, Any]:
    metric_variances: dict[str, float] = {}
    for metric in metrics:
        values = coerce_numeric_series(metric_series.get(metric, ()))
        if len(values) < 2:
            metric_variances[metric] = 0.0
            continue
        mu = mean(values)
        var = sum((v - mu) ** 2 for v in values) / len(values)
        metric_variances[metric] = round(var, 6)

    total = sum(metric_variances.values())
    shares: dict[str, float] = {}
    for metric, var in metric_variances.items():
        share = (var / total) if total > 0.0 else 0.0
        shares[metric] = round(share, 6)

    return {
        "total_variance": round(total, 6),
        "metric_variances": metric_variances,
        "variance_share": shares,
    }


def _segment_contribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        raw_segment = row.get("segment")
        if raw_segment is None:
            metadata = row.get("metadata_json")
            if isinstance(metadata, Mapping):
                raw_segment = metadata.get("segment") or metadata.get("customer_segment")
        segment = str(raw_segment or "").strip()
        if not segment:
            continue

        value = row.get("metric_value")
        if isinstance(value, Mapping):
            value = value.get("value")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric):
            continue
        buckets[segment] = buckets.get(segment, 0.0) + float(numeric)

    total = sum(max(0.0, val) for val in buckets.values())
    contributors: list[dict[str, Any]] = []
    for segment, value in sorted(buckets.items(), key=lambda item: (-item[1], item[0])):
        share = (max(0.0, value) / total) if total > 0.0 else 0.0
        contributors.append(
            {
                "segment": segment,
                "contribution_value": round(value, 6),
                "contribution_share": round(share, 6),
            }
        )

    return {
        "total_value": round(total, 6),
        "segments": contributors,
        "top_segment": contributors[0] if contributors else None,
    }
