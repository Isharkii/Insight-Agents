"""
agent/nodes/signal_conflict_node.py

Global semantic signal conflict detection.

Runs after insight generation stages and before risk/reasoning layers.
"""

from __future__ import annotations

from typing import Any, Mapping

from agent.helpers.kpi_extraction import metric_series_from_kpi_payload, resolve_kpi_payload
from agent.nodes.node_result import confidence_of, payload_of, status_of, success
from agent.signal_integrity import UnifiedSignalIntegrity
from agent.signal_normalizer import normalize_forecast_signals, normalize_kpi_signals
from agent.state import AgentState
from app.services.statistics.leading_indicators import detect_leading_indicators
from app.services.statistics.signal_conflict import (
    apply_conflict_penalty,
    detect_conflicts,
)


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _series_direction(values: list[float]) -> int:
    if len(values) < 2:
        return 0
    delta = values[-1] - values[0]
    if delta > 0:
        return 1
    if delta < 0:
        return -1
    return 0


def _temporal_conflicts_from_leading(
    metric_series: Mapping[str, list[float]],
    leading_result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    relationships = leading_result.get("relationships")
    if not isinstance(relationships, list):
        return []

    conflicts: list[dict[str, Any]] = []
    for item in relationships:
        if not isinstance(item, Mapping):
            continue
        if not bool(item.get("significant")):
            continue
        lag = item.get("optimal_lag")
        if not isinstance(lag, int) or lag == 0:
            continue
        metric_a = str(item.get("metric_a") or "").strip()
        metric_b = str(item.get("metric_b") or "").strip()
        if not metric_a or not metric_b:
            continue
        dir_a = _series_direction(metric_series.get(metric_a, []))
        dir_b = _series_direction(metric_series.get(metric_b, []))
        if dir_a == 0 or dir_b == 0:
            continue
        if dir_a != dir_b:
            conflicts.append(
                {
                    "metric_a": metric_a,
                    "metric_b": metric_b,
                    "optimal_lag": lag,
                    "relationship": item.get("relationship"),
                    "optimal_correlation": item.get("optimal_correlation"),
                    "direction_a": "up" if dir_a > 0 else "down",
                    "direction_b": "up" if dir_b > 0 else "down",
                    "reason": "divergent_temporal_direction",
                }
            )
    return conflicts


def _extract_forecast_signals_direct(
    forecast_payload: dict[str, Any],
    signals: dict[str, float],
) -> None:
    """Extract slope and deviation directly from forecast data.

    Fallback when normalize_forecast_signals raises (e.g. fallback forecasts
    with non-standard structure). Iterates all metric rows and uses the first
    non-None values found.
    """
    forecasts = forecast_payload.get("forecasts")
    if not isinstance(forecasts, Mapping):
        return

    for _metric, row in forecasts.items():
        if not isinstance(row, Mapping):
            continue
        data = row.get("forecast_data")
        if not isinstance(data, Mapping):
            continue

        if "slope" not in signals:
            slope = _safe_float(data.get("slope"))
            if slope is not None:
                signals["slope"] = slope

        if "deviation_percentage" not in signals:
            deviation = _safe_float(data.get("deviation_percentage"))
            if deviation is not None:
                signals["deviation_percentage"] = deviation

        # Also extract r_squared as a confidence signal
        r_sq = _safe_float(data.get("r_squared"))
        if r_sq is not None and "forecast_r_squared" not in signals:
            signals["forecast_r_squared"] = r_sq

    # Extract confidence_score from confidence_breakdown if available
    breakdown = forecast_payload.get("confidence_breakdown")
    if isinstance(breakdown, Mapping):
        fc = _safe_float(breakdown.get("confidence_score"))
        if fc is not None and "forecast_confidence" not in signals:
            signals["forecast_confidence"] = fc


def _collect_global_signals(
    state: AgentState,
) -> tuple[dict[str, float], list[str], dict[str, Any]]:
    signals: dict[str, float] = {}
    warnings: list[str] = []
    metadata: dict[str, Any] = {"leading_indicators": {}, "temporal_conflicts": []}

    # ── Isolation enforcement ─────────────────────────────────────
    # Compute layer integrity to identify isolated signals.
    # Isolated layers must NOT contribute signals to conflict detection —
    # otherwise an isolated forecast (low R²) can generate spurious conflicts
    # that cascade into risk scoring and prioritization.
    try:
        _integrity = UnifiedSignalIntegrity.compute(state)
        _isolated = set(_integrity.get("isolated_layers", []))
    except Exception:
        _isolated = set()

    if _isolated:
        import logging as _iso_logging
        _iso_logging.getLogger("agent.nodes.signal_conflict").info(
            "Isolation enforcement: excluding signals from layers: %s",
            ", ".join(sorted(_isolated)),
        )
        metadata["isolated_layers"] = sorted(_isolated)

    kpi_payload = resolve_kpi_payload(state)
    if isinstance(kpi_payload, Mapping) and kpi_payload:
        metric_series = metric_series_from_kpi_payload(kpi_payload)
        if metric_series:
            leading = detect_leading_indicators(metric_series)
            metadata["leading_indicators"] = leading
            warnings.extend(
                str(item)
                for item in leading.get("warnings", [])
                if str(item).strip()
            )
            temporal_conflicts = _temporal_conflicts_from_leading(metric_series, leading)
            metadata["temporal_conflicts"] = temporal_conflicts
            if temporal_conflicts:
                warnings.append(
                    f"Detected {len(temporal_conflicts)} temporal leading/lagging divergence(s)."
                )
        try:
            kpi_signals = normalize_kpi_signals(dict(kpi_payload), strict=False)
            raw_warnings = kpi_signals.pop("_warnings", [])
            if isinstance(raw_warnings, list):
                warnings.extend(str(item) for item in raw_warnings if str(item).strip())
            # Exclude defaulted (phantom-zero) signals entirely — they
            # represent absent data, not real "no change" observations.
            defaulted = kpi_signals.pop("_defaulted", [])
            if not isinstance(defaulted, list):
                defaulted = []
            defaulted_set = set(defaulted)
            for key, value in kpi_signals.items():
                if str(key) in defaulted_set:
                    continue
                numeric = _safe_float(value)
                if numeric is not None:
                    signals[str(key)] = numeric
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"kpi_signal_normalization_failed: {exc}")

    # ── Forecast signals (skip if layer is isolated) ─────────────
    if "forecast" in _isolated:
        warnings.append(
            "forecast signals excluded from conflict detection: layer isolated"
        )
    else:
        forecast_envelope = state.get("forecast_data")
        if status_of(forecast_envelope) == "success":
            forecast_payload = payload_of(forecast_envelope) or {}
            # Always extract slope and deviation independently first — these
            # are the primary signals for conflict detection.  The bundled
            # normalize_forecast_signals() derives churn_acceleration which
            # frequently throws (no churn forecast row), killing all three
            # signals.  By extracting the two critical signals up-front we
            # guarantee they reach the conflict engine.
            _extract_forecast_signals_direct(forecast_payload, signals)
            try:
                forecast_signals = normalize_forecast_signals(forecast_payload)
                for key, value in forecast_signals.items():
                    numeric = _safe_float(value)
                    if numeric is not None:
                        signals.setdefault(str(key), numeric)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"forecast_signal_normalization_failed: {exc}")

    growth_payload = payload_of(state.get("growth_data")) or {}
    horizons = growth_payload.get("primary_horizons")
    if isinstance(horizons, Mapping):
        short_growth = _safe_float(horizons.get("short_growth"))
        mid_growth = _safe_float(horizons.get("mid_growth"))
        long_growth = _safe_float(horizons.get("long_growth"))
        if "revenue_growth_delta" not in signals:
            for candidate in (short_growth, mid_growth, long_growth):
                if candidate is not None:
                    signals["revenue_growth_delta"] = candidate
                    break
        trend_acc = _safe_float(horizons.get("trend_acceleration"))
        if trend_acc is not None:
            signals.setdefault("forecast_slope", trend_acc)

    cohort_payload = payload_of(state.get("cohort_data")) or {}
    cohort_signals = cohort_payload.get("signals")
    if isinstance(cohort_signals, Mapping):
        churn_acc = _safe_float(cohort_signals.get("churn_acceleration"))
        if churn_acc is not None:
            signals.setdefault("churn_delta", churn_acc)
        retention_decay = _safe_float(cohort_signals.get("retention_decay"))
        if retention_decay is not None:
            signals.setdefault("active_customer_delta", -retention_decay)

    multivariate_payload = payload_of(state.get("multivariate_scenario_data")) or {}
    scenario = multivariate_payload.get("scenario_simulation")
    if isinstance(scenario, Mapping):
        scenarios = scenario.get("scenarios")
        if isinstance(scenarios, Mapping):
            best = scenarios.get("best") if isinstance(scenarios.get("best"), Mapping) else {}
            worst = scenarios.get("worst") if isinstance(scenarios.get("worst"), Mapping) else {}
            best_growth = _safe_float(best.get("projected_growth"))
            worst_growth = _safe_float(worst.get("projected_growth"))
            if best_growth is not None and worst_growth is not None:
                signals.setdefault("deviation_percentage", abs(best_growth - worst_growth))
                signals.setdefault("slope", worst_growth)

    unit_payload = payload_of(state.get("unit_economics_data")) or {}
    trends = unit_payload.get("trends")
    if isinstance(trends, Mapping):
        cac_delta = _safe_float(trends.get("cac"))
        ltv_delta = _safe_float(trends.get("ltv"))
        if cac_delta is not None:
            signals.setdefault("cac_delta", cac_delta)
        if ltv_delta is not None:
            signals.setdefault("ltv_delta", ltv_delta)

    # Extract growth efficiency burn risk as a synthetic signal
    growth_efficiency = unit_payload.get("growth_efficiency")
    if isinstance(growth_efficiency, Mapping):
        velocity = _safe_float(growth_efficiency.get("revenue_velocity"))
        if velocity is not None:
            signals.setdefault("revenue_growth_delta", velocity)

    return signals, warnings, metadata


def signal_conflict_node(state: AgentState) -> AgentState:
    """Compute global signal conflicts and write envelope to state."""
    signals, warnings, metadata = _collect_global_signals(state)
    conflict_result = detect_conflicts(signals)

    upstream_keys = (
        "growth_data",
        "timeseries_factors_data",
        "cohort_data",
        "category_formula_data",
        "unit_economics_data",
        "multivariate_scenario_data",
        "segmentation",
        "forecast_data",
    )
    upstream_conf = [
        confidence_of(state.get(key))
        for key in upstream_keys
        if isinstance(state.get(key), Mapping)
    ]
    base_confidence = min(upstream_conf) if upstream_conf else 1.0
    adjustment = apply_conflict_penalty(
        base_confidence=base_confidence,
        conflict_result=conflict_result,
        floor=0.0,
    )

    temporal_conflicts = metadata.get("temporal_conflicts", [])
    temporal_count = len(temporal_conflicts) if isinstance(temporal_conflicts, list) else 0
    temporal_penalty = min(0.18, 0.04 * float(temporal_count))
    if temporal_penalty > 0.0:
        adjusted = max(
            0.0,
            float(adjustment.get("adjusted_confidence", 0.0)) - temporal_penalty,
        )
        adjustment["adjusted_confidence"] = round(adjusted, 6)
        adjustment["temporal_penalty"] = round(temporal_penalty, 6)
        adjustment["adjustment_reason"] = (
            f"{adjustment.get('adjustment_reason', 'conflicts')} + temporal_penalty({temporal_penalty:.3f})"
        )
        conflict_result["temporal_conflict_count"] = temporal_count
        conflict_result["confidence_penalty"] = round(
            float(conflict_result.get("confidence_penalty", 0.0)) + temporal_penalty,
            6,
        )
    else:
        conflict_result["temporal_conflict_count"] = temporal_count

    node_warnings = list(warnings)
    node_warnings.extend(str(item) for item in conflict_result.get("warnings", []))

    payload = {
        "signals_used": signals,
        "conflict_result": conflict_result,
        "confidence_adjustment": adjustment,
        "leading_indicators": metadata.get("leading_indicators", {}),
        "temporal_conflicts": temporal_conflicts if isinstance(temporal_conflicts, list) else [],
    }
    envelope = success(
        payload,
        warnings=node_warnings,
        confidence_score=float(adjustment.get("adjusted_confidence", 0.0)),
    )
    return {"signal_conflicts": envelope}
