"""
agent/nodes/timeseries_factors_node.py

Compute deterministic time-series factor flags (momentum, volatility, etc.).
"""

from __future__ import annotations

from agent.helpers.confidence_model import compute_standard_confidence
from agent.helpers.kpi_extraction import (
    dataset_confidence_from_state,
    metric_series_from_kpi_payload,
    records_from_kpi_payload,
    resolve_kpi_payload,
)
from agent.nodes.node_result import failed, payload_of, skipped, success
from agent.state import AgentState
from app.services.timeseries_factors import compute_timeseries_factors
from app.services.statistics.changepoint import detect_changepoints


def timeseries_factors_node(state: AgentState) -> AgentState:
    """Compute deterministic time-series factor flags."""
    try:
        kpi_payload = resolve_kpi_payload(state)
        records = records_from_kpi_payload(kpi_payload)
        metric_series = metric_series_from_kpi_payload(kpi_payload)
        if not metric_series:
            return {
                "timeseries_factors_data": skipped(
                    "series_unavailable",
                    {"records": len(records), "record_count": int(kpi_payload.get("record_count") or 0)},
                ),
            }

        growth_payload = payload_of(state.get("growth_data")) or {}
        primary_metric = str(growth_payload.get("primary_metric") or "").strip()
        if primary_metric not in metric_series:
            primary_metric = sorted(metric_series, key=lambda name: (-len(metric_series[name]), name))[0]

        primary_values = metric_series.get(primary_metric, [])
        factors = compute_timeseries_factors(primary_values)
        changepoint_result = detect_changepoints(primary_values)
        factors["changepoints"] = changepoint_result
        warnings: list[str] = []
        if str(factors.get("volatility_regime") or "") == "insufficient_history":
            warnings.append("Insufficient history for volatility regime detection.")
        if str(factors.get("cycle_state") or "") == "insufficient_history":
            warnings.append("Insufficient history for cycle state detection.")
        seasonality = factors.get("seasonality")
        if isinstance(seasonality, dict):
            for w in seasonality.get("warnings", []):
                warnings.append(str(w))
        for w in changepoint_result.get("warnings", []):
            warnings.append(str(w))

        dataset_confidence = dataset_confidence_from_state(state)
        volatility_regime = str(factors.get("volatility_regime") or "").strip().lower()
        cycle_state = str(factors.get("cycle_state") or "").strip().lower()
        seasonality_strength = None
        if isinstance(seasonality, dict):
            seasonality_strength = seasonality.get("strength")
        upstream_growth_confidence = 1.0
        growth_envelope = state.get("growth_data")
        if isinstance(growth_envelope, dict):
            raw_growth_confidence = growth_envelope.get("confidence_score")
            if isinstance(raw_growth_confidence, (int, float)):
                upstream_growth_confidence = float(raw_growth_confidence)
        confidence_model = compute_standard_confidence(
            values=primary_values,
            signals={
                "momentum_up": 1.0 if bool(factors.get("momentum_up")) else -1.0,
                "momentum_down": -1.0 if bool(factors.get("momentum_down")) else 1.0,
                "volatility_regime_signal": (
                    -1.0
                    if volatility_regime == "high"
                    else (1.0 if volatility_regime == "low" else 0.0)
                ),
                "structural_break_signal": (
                    -1.0 if bool(factors.get("structural_break_detected")) else 1.0
                ),
                "cycle_signal": (
                    1.0
                    if cycle_state in {"expansion", "trough"}
                    else (-1.0 if cycle_state in {"contraction", "peak"} else 0.0)
                ),
                "seasonality_strength": seasonality_strength,
                "changepoint_count": (
                    -float(
                        (
                            changepoint_result.get("summary", {})
                            if isinstance(changepoint_result.get("summary"), dict)
                            else {}
                        ).get("changepoints_detected", 0)
                    )
                ),
                "changepoint_methods_agreeing": (
                    (
                        changepoint_result.get("summary", {})
                        if isinstance(changepoint_result.get("summary"), dict)
                        else {}
                    ).get("methods_agreeing", 0)
                ),
            },
            dataset_confidence=dataset_confidence,
            upstream_confidences=[upstream_growth_confidence],
            status="partial" if warnings else "success",
            base_warnings=warnings,
        )
        payload = {
            "primary_metric": primary_metric,
            "series_points": len(metric_series.get(primary_metric, [])),
            "factors": factors,
            "confidence_breakdown": confidence_model,
        }
        return {
            "timeseries_factors_data": success(
                payload,
                warnings=confidence_model["warnings"],
                confidence_score=float(confidence_model["confidence_score"]),
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "timeseries_factors_data": failed(str(exc), {"stage": "timeseries_factors"}),
        }
