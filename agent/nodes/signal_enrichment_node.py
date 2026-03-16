"""
agent/nodes/signal_enrichment_node.py

Signal Enrichment Node: normalizes all upstream analytical signals into
a clean, human-readable structure for the LLM synthesis layer.

Runs immediately before synthesis_gate. The LLM reads this enriched
signal structure instead of raw node payloads, enabling higher quality
narratives without exposing internal envelope details.

This node is purely deterministic — no LLM, no I/O, no side-effects.
"""

from __future__ import annotations

import logging
from typing import Any

from agent.helpers.kpi_extraction import (
    metric_series_from_kpi_payload,
    resolve_kpi_payload,
)
from agent.nodes.node_result import confidence_of, payload_of, status_of, success
from agent.state import AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------


def _classify_trend(values: list[float]) -> str:
    """Classify a numeric series as accelerating, stable, or declining."""
    if len(values) < 2:
        return "unknown"

    # Use last half vs first half comparison
    mid = len(values) // 2
    first_half_avg = sum(values[:mid]) / max(1, mid)
    second_half_avg = sum(values[mid:]) / max(1, len(values) - mid)

    if first_half_avg == 0:
        return "stable" if second_half_avg == 0 else "accelerating"

    pct_change = (second_half_avg - first_half_avg) / abs(first_half_avg)

    if pct_change > 0.05:
        return "accelerating"
    elif pct_change < -0.05:
        return "declining"
    return "stable"


def _classify_volatility(values: list[float]) -> str:
    """Classify a series' volatility as low, medium, or high."""
    if len(values) < 3:
        return "unknown"

    mean = sum(values) / len(values)
    if mean == 0:
        return "low"

    variance = sum((v - mean) ** 2 for v in values) / len(values)
    cv = (variance ** 0.5) / abs(mean)  # coefficient of variation

    if cv > 0.3:
        return "high"
    elif cv > 0.1:
        return "medium"
    return "low"


def _classify_forecast_direction(forecast_payload: dict[str, Any] | None) -> str:
    """Determine forecast direction from forecast node payload."""
    if not forecast_payload:
        return "unknown"

    forecasts = forecast_payload.get("forecasts")
    if not isinstance(forecasts, dict):
        return "unknown"

    slopes: list[float] = []
    for _metric, row in forecasts.items():
        if not isinstance(row, dict):
            continue
        data = row.get("forecast_data")
        if not isinstance(data, dict):
            continue
        slope = data.get("slope")
        if isinstance(slope, (int, float)):
            slopes.append(float(slope))

    if not slopes:
        return "unknown"

    avg_slope = sum(slopes) / len(slopes)
    if avg_slope > 0.01:
        return "upward"
    elif avg_slope < -0.01:
        return "downward"
    return "flat"


def _classify_cohort_health(cohort_payload: dict[str, Any] | None) -> str:
    """Determine cohort health from cohort node payload."""
    if not cohort_payload:
        return "unknown"

    retention_decay = cohort_payload.get("retention_decay")
    if retention_decay is None:
        signals = cohort_payload.get("signals")
        if isinstance(signals, dict):
            retention_decay = signals.get("retention_decay")

    if not isinstance(retention_decay, (int, float)):
        return "unknown"

    decay = float(retention_decay)
    if decay < 0.2:
        return "strong"
    elif decay < 0.5:
        return "moderate"
    return "weakening"


def _extract_primary_risk(risk_payload: dict[str, Any] | None) -> str:
    """Extract the primary risk signal from risk node payload."""
    if not risk_payload:
        return "no risk data available"

    # Try structured risk categories
    categories = risk_payload.get("risk_categories")
    if isinstance(categories, list) and categories:
        # Find highest severity risk
        for cat in sorted(
            categories,
            key=lambda c: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(
                str(c.get("severity", "")).lower(), 4
            ),
        ):
            description = cat.get("description") or cat.get("category") or cat.get("name")
            if description:
                return str(description)

    # Try top-level risk summary
    for key in ("primary_risk", "risk_summary", "overall_risk", "risk_level"):
        value = risk_payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return "risk assessment completed without critical findings"


def _extract_growth_signals(growth_payload: dict[str, Any] | None) -> dict[str, Any]:
    """Extract key growth signals for enrichment."""
    if not growth_payload:
        return {}

    signals: dict[str, Any] = {}
    for key in ("growth_rate", "growth_trend", "momentum", "acceleration"):
        value = growth_payload.get(key)
        if value is not None:
            signals[key] = value

    # Include numeric horizon values so the LLM can cite specific growth rates
    horizons = growth_payload.get("primary_horizons")
    if isinstance(horizons, dict):
        for key in ("short_growth", "mid_growth", "long_growth",
                     "trend_acceleration", "cagr"):
            value = horizons.get(key)
            if value is not None:
                signals[key] = round(float(value), 4) if isinstance(value, (int, float)) else value

    # Primary metric name so LLM knows what was measured
    primary_metric = growth_payload.get("primary_metric")
    if primary_metric:
        signals["primary_metric"] = str(primary_metric)

    return signals


def _extract_key_metrics(
    metric_series: dict[str, list[float]],
    forecast_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """Pre-compute concrete metric summaries the LLM can cite directly.

    Returns a dict with actual values, deltas, and period-over-period changes
    so the LLM doesn't need to compute anything — just reference numbers.
    """
    metrics: dict[str, Any] = {}

    for name, values in metric_series.items():
        if not values or len(values) < 2:
            continue
        latest = values[-1]
        previous = values[-2]
        first = values[0]
        delta_pct = ((latest - previous) / abs(previous) * 100) if previous != 0 else 0
        total_change_pct = ((latest - first) / abs(first) * 100) if first != 0 else 0
        metrics[name] = {
            "latest_value": round(latest, 2),
            "previous_value": round(previous, 2),
            "first_value": round(first, 2),
            "period_change_pct": round(delta_pct, 2),
            "total_change_pct": round(total_change_pct, 2),
            "data_points": len(values),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
        }

    # Add forecast-specific numbers
    if forecast_payload:
        forecasts = forecast_payload.get("forecasts")
        if isinstance(forecasts, dict):
            for metric_name, row in forecasts.items():
                if not isinstance(row, dict):
                    continue
                data = row.get("forecast_data")
                if not isinstance(data, dict):
                    continue
                forecast_info: dict[str, Any] = {}
                for key in ("slope", "deviation_percentage", "r_squared",
                            "confidence_score"):
                    val = data.get(key)
                    if isinstance(val, (int, float)):
                        forecast_info[key] = round(float(val), 4)
                regression = data.get("regression")
                if isinstance(regression, dict):
                    r2 = regression.get("r_squared")
                    if isinstance(r2, (int, float)):
                        forecast_info["r_squared"] = round(float(r2), 4)
                if forecast_info:
                    entry = metrics.get(str(metric_name), {})
                    entry["forecast"] = forecast_info
                    metrics[str(metric_name)] = entry

    return metrics


def _compute_signal_confidence(state: AgentState) -> float:
    """Compute an aggregate signal confidence from all upstream envelopes."""
    envelope_keys = [
        "kpi_data", "saas_kpi_data", "ecommerce_kpi_data", "agency_kpi_data",
        "forecast_data", "cohort_data", "growth_data",
        "timeseries_factors_data", "risk_data",
    ]
    confidences: list[float] = []
    for key in envelope_keys:
        envelope = state.get(key)
        if envelope is not None and status_of(envelope) in ("success", "insufficient_data"):
            conf = confidence_of(envelope)
            if conf > 0:
                confidences.append(conf)

    if not confidences:
        return 0.0

    return round(sum(confidences) / len(confidences), 6)


def signal_enrichment_node(state: AgentState) -> AgentState:
    """LangGraph node: normalize upstream signals into enriched structure.

    Reads all upstream analytical envelopes and produces a clean
    signal summary that the LLM can reason from directly.

    Writes:
        state["signal_enrichment"] — dict with classified signals.
    """
    # Extract KPI-level growth trend
    kpi_payload = resolve_kpi_payload(state)
    metric_series = metric_series_from_kpi_payload(kpi_payload) if kpi_payload else {}

    # Find the primary revenue metric
    revenue_candidates = (
        "revenue", "mrr", "arr", "total_revenue", "recurring_revenue",
        "net_revenue", "gross_revenue", "sales", "value", "timeseries_value",
    )
    growth_trend = "unknown"
    volatility_level = "unknown"
    for candidate in revenue_candidates:
        series = metric_series.get(candidate)
        if series and len(series) >= 2:
            growth_trend = _classify_trend(series)
            volatility_level = _classify_volatility(series)
            break

    # If no revenue metric, try any available metric
    if growth_trend == "unknown" and metric_series:
        first_key = next(iter(metric_series))
        series = metric_series[first_key]
        if len(series) >= 2:
            growth_trend = _classify_trend(series)
            volatility_level = _classify_volatility(series)

    # Forecast direction
    forecast_payload = payload_of(state.get("forecast_data"))
    forecast_direction = _classify_forecast_direction(forecast_payload)

    # Cohort health
    cohort_payload = payload_of(state.get("cohort_data"))
    cohort_health = _classify_cohort_health(cohort_payload)

    # Primary risk
    risk_payload = payload_of(state.get("risk_data"))
    primary_risk = _extract_primary_risk(risk_payload)

    # Growth signals
    growth_payload = payload_of(state.get("growth_data"))
    growth_signals = _extract_growth_signals(growth_payload)

    # Key metrics with concrete numbers the LLM can cite
    key_metrics = _extract_key_metrics(metric_series, forecast_payload)

    # Unit economics signals
    ue_payload = payload_of(state.get("unit_economics_data"))
    unit_economics_summary: dict[str, Any] = {"status": "unavailable"}
    if ue_payload:
        ue_metrics = ue_payload.get("metrics")
        if isinstance(ue_metrics, dict):
            unit_economics_summary = {
                "status": "available",
                "ltv": ue_metrics.get("ltv"),
                "cac": ue_metrics.get("cac"),
                "ltv_cac_ratio": ue_metrics.get("ltv_cac_ratio"),
                "churn_rate": ue_metrics.get("churn_rate"),
                "burn_risk": (
                    ue_payload.get("growth_efficiency", {}).get("burn_risk_indicator")
                    if isinstance(ue_payload.get("growth_efficiency"), dict)
                    else None
                ),
                "estimation_method": ue_payload.get("estimation_method", "explicit"),
                "signal_summary": ue_payload.get("signal_summary"),
            }

    # Aggregate confidence
    signal_confidence = _compute_signal_confidence(state)

    # Data source annotations
    forecast_source = "unknown"
    if forecast_payload:
        forecast_source = forecast_payload.get("forecast_source", "db_model")

    cohort_method = "unknown"
    if cohort_payload:
        cohort_method = cohort_payload.get("method", "standard_cohort")

    ue_method = "unknown"
    if ue_payload:
        ue_method = ue_payload.get("estimation_method", "explicit")

    enrichment: dict[str, Any] = {
        "growth_trend": growth_trend,
        "volatility_level": volatility_level,
        "forecast_direction": forecast_direction,
        "cohort_health": cohort_health,
        "primary_risk": primary_risk,
        "unit_economics": unit_economics_summary,
        "signal_confidence": signal_confidence,
        "data_sources": {
            "forecast_source": forecast_source,
            "cohort_method": cohort_method,
            "unit_economics_method": ue_method,
        },
        "growth_signals": growth_signals,
        "key_metrics": key_metrics,
        "available_metrics": list(metric_series.keys()),
        "metric_count": len(metric_series),
    }

    logger.info(
        "Signal enrichment: trend=%s volatility=%s forecast=%s cohort=%s confidence=%.2f",
        growth_trend,
        volatility_level,
        forecast_direction,
        cohort_health,
        signal_confidence,
    )

    return {
        "signal_enrichment": success(
            enrichment,
            confidence_score=signal_confidence,
        ),
    }
