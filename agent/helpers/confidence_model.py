"""
agent/helpers/confidence_model.py

Standardized confidence interface for insight pipeline nodes.

All node-level confidence is derived from:
    1) Statistical model score from ``compute_confidence()``
    2) Dataset confidence cap
    3) Upstream confidence propagation cap
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from app.services.statistics.confidence_scoring import compute_confidence


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def clamp01(value: Any, *, default: float = 0.0) -> float:
    parsed = _safe_float(value)
    if parsed is None:
        parsed = default
    return max(0.0, min(1.0, float(parsed)))


def coerce_series(values: Sequence[Any]) -> list[float]:
    out: list[float] = []
    for raw in values:
        parsed = _safe_float(raw)
        if parsed is not None:
            out.append(parsed)
    return out


def coerce_signal_map(signals: Mapping[str, Any] | None) -> dict[str, float | None]:
    if not isinstance(signals, Mapping):
        return {}
    out: dict[str, float | None] = {}
    for key, raw in signals.items():
        name = str(key).strip()
        if not name:
            continue
        out[name] = _safe_float(raw)
    return out


def propagated_confidence(
    *,
    model_confidence: float,
    dataset_confidence: float = 1.0,
    upstream_confidences: Sequence[float] = (),
    isolated_upstream_indices: Sequence[int] = (),
) -> tuple[float, dict[str, float]]:
    """Propagate confidence through the pipeline.

    Uses a **weighted-average** approach instead of strict ``min()``.
    The model's own confidence carries 60% weight; dataset and upstream
    confidences contribute the remaining 40%.  This prevents a single
    weak upstream node from crushing an otherwise strong signal.

    Parameters
    ----------
    isolated_upstream_indices:
        Indices into ``upstream_confidences`` that should be excluded from
        propagation because those signals have been isolated (broken module,
        R² too low, etc.).  Isolated signals must not influence healthy ones.
    """
    model_conf = clamp01(model_confidence)
    dataset_cap = clamp01(dataset_confidence, default=1.0)

    # Filter out isolated upstream signals before averaging.
    isolated_set = set(isolated_upstream_indices)
    active_upstream = [
        clamp01(value, default=1.0)
        for i, value in enumerate(upstream_confidences)
        if i not in isolated_set
    ]

    upstream_cap = (
        sum(active_upstream) / len(active_upstream)
        if active_upstream
        else 1.0
    )

    # Weighted blend: model 60%, dataset 20%, upstream 20%.
    propagation_cap = 0.6 * model_conf + 0.2 * dataset_cap + 0.2 * upstream_cap

    # Still respect a hard floor from truly broken upstream: if ANY active
    # upstream is below 0.15, apply a moderate penalty rather than zeroing out.
    upstream_min = min(active_upstream) if active_upstream else 1.0
    if upstream_min < 0.15:
        propagation_cap = min(propagation_cap, 0.4)

    final = round(max(0.0, min(1.0, propagation_cap)), 6)
    return (
        final,
        {
            "model_confidence": round(model_conf, 6),
            "dataset_cap": round(dataset_cap, 6),
            "upstream_cap": round(upstream_cap, 6),
            "upstream_min": round(upstream_min, 6),
            "propagation_cap": round(propagation_cap, 6),
            "isolated_count": len(isolated_set),
            "active_upstream_count": len(active_upstream),
        },
    )


def compute_standard_confidence(
    *,
    values: Sequence[Any],
    signals: Mapping[str, Any] | None = None,
    dataset_confidence: float = 1.0,
    upstream_confidences: Sequence[float] = (),
    isolated_upstream_indices: Sequence[int] = (),
    tier_cap: float | None = None,
    status: str = "success",
    base_warnings: Sequence[str] = (),
) -> dict[str, Any]:
    series = coerce_series(values)
    normalized_signals = coerce_signal_map(signals)
    model_result = compute_confidence(
        series,
        signals=normalized_signals,
        tier_cap=tier_cap,
    )
    model_conf = clamp01(model_result.get("confidence_score"), default=0.0)
    final_conf, propagation = propagated_confidence(
        model_confidence=model_conf,
        dataset_confidence=dataset_confidence,
        upstream_confidences=upstream_confidences,
        isolated_upstream_indices=isolated_upstream_indices,
    )

    normalized_status = str(status or "").strip().lower()
    if normalized_status == "insufficient_data":
        # Penalize rather than hard-cap: reduce by 20% but allow up to 0.65.
        # This preserves signal strength while still reflecting data gaps.
        final_conf = min(final_conf * 0.8, 0.65)
    elif normalized_status in {"failed", "skipped"}:
        final_conf = 0.0

    warnings: list[str] = []
    warnings.extend(str(item) for item in base_warnings if str(item).strip())
    model_warnings = model_result.get("warnings")
    if isinstance(model_warnings, list):
        warnings.extend(str(item) for item in model_warnings if str(item).strip())

    return {
        "confidence_score": round(final_conf, 6),
        "model_confidence_score": round(model_conf, 6),
        "components": list(model_result.get("components") or []),
        "formula": str(model_result.get("formula") or ""),
        "warnings": warnings,
        "signals": normalized_signals,
        "propagation": propagation,
        "tier_cap": tier_cap,
        "status": normalized_status or "success",
    }


def propagate_reasoning_strategy_confidence(
    *,
    insight_confidence: float,
    reasoning_confidence: float | None = None,
    strategy_penalty: float = 0.0,
) -> dict[str, float]:
    insight = clamp01(insight_confidence, default=0.0)
    reasoning = clamp01(reasoning_confidence, default=insight)
    strategy = max(0.0, min(1.0, reasoning - max(0.0, float(strategy_penalty))))
    return {
        "insight_confidence": round(insight, 6),
        "reasoning_confidence": round(reasoning, 6),
        "strategy_confidence": round(strategy, 6),
    }
