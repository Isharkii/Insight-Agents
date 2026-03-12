"""
agent/nodes/forecast_node.py

Forecast Fetch Node: retrieves the latest persisted forecast for each
relevant metric of the entity named in state.

When no persisted forecasts exist, computes a deterministic fallback
forecast from KPI history using linear regression (>= 6 points) or
simple trend estimation (< 6 points).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from agent.helpers.confidence_model import compute_standard_confidence
from agent.helpers.kpi_extraction import (
    dataset_confidence_from_state,
    metric_series_from_kpi_payload,
    resolve_kpi_payload,
)
from agent.nodes.node_result import failed, skipped, success
from agent.state import AgentState
from db.session import SessionLocal
from forecast.repository import ForecastRepository

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metric sets per business type
# ---------------------------------------------------------------------------

_DEFAULT_METRICS: list[str] = ["revenue", "growth_rate", "sales"]

_METRICS_BY_BUSINESS_TYPE: dict[str, list[str]] = {
    "saas": ["mrr", "churn_rate", "ltv", "growth_rate"],
    "ecommerce": [
        "revenue",
        "aov",
        "conversion_rate",
        "cac",
        "purchase_frequency",
        "ltv",
        "growth_rate",
    ],
    "agency": [
        "total_revenue",
        "client_churn",
        "utilization_rate",
        "revenue_per_employee",
        "client_ltv",
    ],
}

# Confidence penalty applied when using fallback forecasts instead of
# persisted model forecasts.
_FALLBACK_FORECAST_PENALTY = -0.1

# Minimum datapoints for full linear regression; below this, use simple trend.
_MIN_REGRESSION_POINTS = 6

# Number of future periods to forecast.
_FORECAST_HORIZON = 3


def _metrics_for(business_type: str) -> list[str]:
    return _METRICS_BY_BUSINESS_TYPE.get(business_type, _DEFAULT_METRICS)


def _serialize_row(row: Any) -> dict[str, Any]:
    """Convert a ForecastMetric ORM row to a plain JSON-safe dict."""
    return {
        "entity_name": row.entity_name,
        "metric_name": row.metric_name,
        "period_end": row.period_end.isoformat(),
        "forecast_data": row.forecast_data,
        "created_at": row.created_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# Fallback forecast computation
# ---------------------------------------------------------------------------


def _linear_regression_forecast(
    values: list[float],
    horizon: int = _FORECAST_HORIZON,
) -> dict[str, Any]:
    """Compute a linear regression forecast from a numeric series.

    Returns a dict with forecast_values, slope, intercept, r_squared,
    model_type, and confidence_score.
    """
    n = len(values)
    x = np.arange(n, dtype=np.float64)
    y = np.array(values, dtype=np.float64)

    # Ordinary least squares
    x_mean = x.mean()
    y_mean = y.mean()
    ss_xx = ((x - x_mean) ** 2).sum()
    ss_xy = ((x - x_mean) * (y - y_mean)).sum()

    if ss_xx == 0:
        # All x values are the same (shouldn't happen with arange)
        return _simple_trend_forecast(values, horizon)

    slope = float(ss_xy / ss_xx)
    intercept = float(y_mean - slope * x_mean)

    # R-squared
    y_pred = slope * x + intercept
    ss_res = float(((y - y_pred) ** 2).sum())
    ss_tot = float(((y - y_mean) ** 2).sum())
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Forecast future periods
    future_x = np.arange(n, n + horizon, dtype=np.float64)
    forecast_values = [round(float(slope * xi + intercept), 6) for xi in future_x]

    # Confidence from R-squared and sample size
    size_factor = min(1.0, n / 12.0)  # 12 months = full confidence
    confidence = round(max(0.0, min(1.0, r_squared * 0.7 + size_factor * 0.3)), 6)

    return {
        "forecast_values": forecast_values,
        "slope": round(slope, 6),
        "intercept": round(intercept, 6),
        "r_squared": round(r_squared, 6),
        "model_type": "linear_trend",
        "confidence_score": confidence,
        "datapoints_used": n,
    }


def _simple_trend_forecast(
    values: list[float],
    horizon: int = _FORECAST_HORIZON,
) -> dict[str, Any]:
    """Compute a simple trend estimate when < 6 datapoints are available.

    Uses the average period-over-period change to project forward.
    """
    n = len(values)
    if n < 2:
        return {
            "forecast_values": [values[0]] * horizon if values else [0.0] * horizon,
            "slope": 0.0,
            "model_type": "insufficient_data",
            "confidence_score": 0.1 if values else 0.0,
            "datapoints_used": n,
        }

    # Average change per period
    changes = [values[i] - values[i - 1] for i in range(1, n)]
    avg_change = sum(changes) / len(changes)
    last_value = values[-1]

    forecast_values = []
    for step in range(1, horizon + 1):
        forecast_values.append(round(last_value + avg_change * step, 6))

    # Low confidence for simple trend
    confidence = round(min(0.4, 0.1 * n), 6)

    return {
        "forecast_values": forecast_values,
        "slope": round(avg_change, 6),
        "model_type": "simple_trend" if n >= 2 else "insufficient_data",
        "confidence_score": confidence,
        "datapoints_used": n,
    }


def _compute_fallback_forecast(
    metric_name: str,
    values: list[float],
) -> dict[str, Any] | None:
    """Compute a deterministic fallback forecast for a single metric.

    Returns None if no values are available.
    """
    if not values:
        return None

    if len(values) >= _MIN_REGRESSION_POINTS:
        forecast = _linear_regression_forecast(values)
    else:
        forecast = _simple_trend_forecast(values)

    forecast["metric"] = metric_name
    return forecast


def _build_fallback_forecasts(
    state: AgentState,
    metrics: list[str],
    entity_name: str,
) -> dict[str, dict[str, Any] | None]:
    """Build fallback forecasts for all requested metrics from KPI history."""
    kpi_payload = resolve_kpi_payload(state)
    if not kpi_payload:
        return {metric: None for metric in metrics}

    metric_series = metric_series_from_kpi_payload(kpi_payload)

    forecasts: dict[str, dict[str, Any] | None] = {}
    for metric in metrics:
        series = metric_series.get(metric)
        if series and len(series) >= 2:
            fallback = _compute_fallback_forecast(metric, series)
            if fallback is not None:
                # Wrap in the same shape as a serialized DB row
                forecasts[metric] = {
                    "entity_name": entity_name,
                    "metric_name": metric,
                    "period_end": "fallback",
                    "forecast_data": fallback,
                    "created_at": "fallback",
                    "source": "fallback",
                }
            else:
                forecasts[metric] = None
        else:
            forecasts[metric] = None

    return forecasts


def forecast_fetch_node(state: AgentState) -> AgentState:
    """
    LangGraph node: fetch the latest forecast records from the repository.

    When no persisted forecasts exist, computes deterministic fallback
    forecasts from KPI history using linear regression or simple trend.

    Reads:
        state["entity_name"]    — entity whose forecasts are fetched.
        state["business_type"]  — determines which metrics to look up.

    Writes:
        state["forecast_data"] — dict with keys:
            "forecasts"      : dict[metric_name, serialised row or None]
            "fetched_for"    : entity_name used for the query
            "metrics_queried": list of metric names attempted
            "error"          : present only on failure
    """
    entity_name: str = state.get("entity_name") or ""
    business_type: str = state.get("business_type") or "general"
    metrics = _metrics_for(business_type)

    try:
        forecasts: dict[str, Any] = {}
        used_fallback = False

        with SessionLocal() as session:
            repo = ForecastRepository(session)
            for metric in metrics:
                row = repo.get_latest_forecast(
                    entity_name=entity_name,
                    metric_name=metric,
                )
                forecasts[metric] = _serialize_row(row) if row else None

        has_any_db_forecast = any(row is not None for row in forecasts.values())

        # Fallback: compute from KPI history when no DB forecasts exist
        if not has_any_db_forecast:
            logger.info(
                "No persisted forecasts for %s; computing fallback from KPI history",
                entity_name,
            )
            fallback_forecasts = _build_fallback_forecasts(state, metrics, entity_name)
            forecasts = fallback_forecasts
            used_fallback = any(row is not None for row in forecasts.values())

        payload: dict[str, Any] = {
            "forecasts": forecasts,
            "fetched_for": entity_name,
            "metrics_queried": metrics,
            "forecast_source": "fallback" if used_fallback else "db_model",
        }

        confidence_values: list[float] = []
        signal_map: dict[str, float | None] = {}
        for metric_name, row in forecasts.items():
            if not isinstance(row, dict):
                continue
            data = row.get("forecast_data")
            if not isinstance(data, dict):
                continue
            raw_conf = data.get("confidence_score")
            if isinstance(raw_conf, (int, float)):
                confidence_values.append(float(raw_conf))
            slope = data.get("slope")
            if isinstance(slope, (int, float)):
                signal_map[f"{metric_name}_slope"] = float(slope)
            deviation = data.get("deviation_percentage")
            if isinstance(deviation, (int, float)):
                signal_map[f"{metric_name}_deviation"] = -abs(float(deviation))
            volatility = data.get("volatility_regime")
            if isinstance(volatility, str):
                vol_norm = volatility.strip().lower()
                signal_map[f"{metric_name}_volatility"] = (
                    -1.0 if vol_norm == "high" else (1.0 if vol_norm == "low" else 0.0)
                )
            # Capture model_type for signal enrichment
            model_type = data.get("model_type")
            if isinstance(model_type, str):
                signal_map[f"{metric_name}_model_type_rank"] = (
                    1.0 if model_type == "linear_trend"
                    else 0.5 if model_type == "simple_trend"
                    else 0.0
                )

        base_warnings: list[str] = []
        if used_fallback:
            base_warnings.append(
                "Forecast computed from KPI history (fallback); "
                "confidence penalty applied."
            )

        has_any_forecast = any(row is not None for row in forecasts.values())
        confidence_model = compute_standard_confidence(
            values=confidence_values,
            signals=signal_map,
            dataset_confidence=dataset_confidence_from_state(state),
            upstream_confidences=[],
            status="success" if confidence_values else "insufficient_data",
            base_warnings=base_warnings,
        )

        # Apply fallback penalty
        if used_fallback:
            raw_conf = float(confidence_model["confidence_score"])
            confidence_model["confidence_score"] = round(
                max(0.0, raw_conf + _FALLBACK_FORECAST_PENALTY), 6,
            )
            confidence_model["fallback_penalty"] = _FALLBACK_FORECAST_PENALTY

        payload["confidence_breakdown"] = confidence_model

        if has_any_forecast:
            forecast_data = success(
                payload,
                warnings=confidence_model["warnings"],
                confidence_score=float(confidence_model["confidence_score"]),
            )
        else:
            forecast_data = skipped("no_forecast_records", payload)

    except Exception as exc:  # noqa: BLE001
        forecast_data = failed(
            str(exc),
            {
                "fetched_for": entity_name,
                "metrics_queried": metrics,
            },
        )

    return {"forecast_data": forecast_data}
