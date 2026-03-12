"""
agent/helpers/statistical_context.py

Deterministic statistical context builder for metric time-series.
Extracted from graph.py to isolate business logic from orchestration.
"""

from __future__ import annotations

from typing import Any, Mapping

from app.services.statistics.anomaly import detect_anomalies
from app.services.statistics.confidence_scoring import compute_confidence
from app.services.statistics.normalization import (
    metric_statistics_config,
    rolling_mean,
    rolling_median,
    zscore_normalize,
)


def build_statistical_context(metric_series: Mapping[str, list[float]]) -> dict[str, Any]:
    """Build per-metric statistical signals (z-score, smoothing, anomaly detection)."""
    if not metric_series:
        return {
            "status": "partial",
            "confidence_score": 0.5,
            "warnings": ["No metric series available for statistical context."],
            "metrics": {},
            "anomaly_summary": {
                "metric_count_with_anomalies": 0,
                "total_anomaly_points": 0,
                "metrics": [],
            },
        }

    metrics_payload: dict[str, Any] = {}
    warnings: list[str] = []
    partial_metrics = 0
    anomaly_metric_names: list[str] = []
    total_anomaly_points = 0
    metric_confidences: list[float] = []

    for metric_name in sorted(metric_series):
        values = metric_series.get(metric_name, [])
        config = metric_statistics_config(metric_name)
        z_values = zscore_normalize(
            values,
            clip_abs=config.zscore_clip,
            zero_guard=config.zero_guard,
        )
        smoothed_mean = rolling_mean(values, window=config.smoothing_window)
        smoothed_median = rolling_median(values, window=config.smoothing_window)
        selected_smoothing = (
            smoothed_median if config.smoothing_method == "median" else smoothed_mean
        )
        anomaly = detect_anomalies(
            values,
            iqr_multiplier=config.anomaly_iqr_multiplier,
        )

        metric_status = "success"
        if len(values) < config.min_points:
            metric_status = "partial"
            partial_metrics += 1
            warnings.append(
                f"Metric '{metric_name}' has {len(values)} points; "
                f"minimum recommended is {config.min_points}."
            )

        anomaly_count = len(anomaly.get("anomaly_indexes", []))
        if anomaly_count > 0:
            anomaly_metric_names.append(metric_name)
            total_anomaly_points += anomaly_count

        confidence_result = compute_confidence(
            values,
            signals={
                "zscore_terminal": z_values[-1] if z_values else 0.0,
                "anomaly_count": -float(anomaly_count),
                "series_length": float(len(values)),
            },
        )
        metric_confidence = float(confidence_result.get("confidence_score") or 0.0)
        metric_confidences.append(metric_confidence)

        metrics_payload[metric_name] = {
            "status": metric_status,
            "series_length": len(values),
            "applied_config": {
                "smoothing_window": config.smoothing_window,
                "smoothing_method": config.smoothing_method,
                "zscore_clip": config.zscore_clip,
                "anomaly_iqr_multiplier": config.anomaly_iqr_multiplier,
                "min_points": config.min_points,
            },
            "zscore": {
                "values": z_values,
                "clip_abs": config.zscore_clip,
            },
            "smoothing": {
                "mean": smoothed_mean,
                "median": smoothed_median,
                "selected_method": config.smoothing_method,
                "selected": selected_smoothing,
            },
            "anomaly": anomaly,
            "confidence_breakdown": confidence_result,
        }

    metric_count = max(1, len(metrics_payload))
    if metric_confidences:
        confidence_score = round(sum(metric_confidences) / len(metric_confidences), 6)
    else:
        confidence_penalty = (partial_metrics / metric_count) * 0.4
        confidence_score = max(0.2, round(1.0 - confidence_penalty, 6))

    return {
        "status": "partial" if partial_metrics > 0 else "success",
        "confidence_score": confidence_score,
        "warnings": warnings,
        "metrics": metrics_payload,
        "anomaly_summary": {
            "metric_count_with_anomalies": len(anomaly_metric_names),
            "total_anomaly_points": total_anomaly_points,
            "metrics": sorted(anomaly_metric_names),
        },
    }
