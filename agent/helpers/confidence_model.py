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
) -> tuple[float, dict[str, float]]:
    dataset_cap = clamp01(dataset_confidence, default=1.0)
    upstream_values = [clamp01(value, default=1.0) for value in upstream_confidences]
    upstream_cap = min(upstream_values) if upstream_values else 1.0
    propagation_cap = min(dataset_cap, upstream_cap)
    final = min(clamp01(model_confidence), propagation_cap)
    return (
        round(final, 6),
        {
            "model_confidence": round(clamp01(model_confidence), 6),
            "dataset_cap": round(dataset_cap, 6),
            "upstream_cap": round(upstream_cap, 6),
            "propagation_cap": round(propagation_cap, 6),
        },
    )


def compute_standard_confidence(
    *,
    values: Sequence[Any],
    signals: Mapping[str, Any] | None = None,
    dataset_confidence: float = 1.0,
    upstream_confidences: Sequence[float] = (),
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
    )

    normalized_status = str(status or "").strip().lower()
    if normalized_status == "insufficient_data":
        final_conf = min(final_conf, 0.45)
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
