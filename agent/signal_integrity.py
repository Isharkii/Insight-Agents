"""Unified deterministic signal integrity model.

This module centralizes confidence math across pipeline nodes.  It replaces
ad-hoc penalty deltas with deterministic layer scores derived from state.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from statistics import mean
from typing import Any, Mapping

from agent.graph_config import KPI_KEY_BY_BUSINESS_TYPE
from agent.nodes.node_result import payload_of, status_of
from agent.state import AgentState

_KPI_MIN_GATE = 0.3
_KPI_FORMULA_WEIGHT = 1.0
_KPI_BACKFILL_WEIGHT = 0.7
_KPI_MIN_DEPTH_POINTS = 3

_FORECAST_MIN_POINTS = 6

_COMPETITIVE_MIN_PEERS = 2
_COMPETITIVE_MIN_METRICS = 2
_COMPETITIVE_SOURCE_RELIABILITY = {
    "local_db": 1.0,
    "deterministic_local": 1.0,
    "external_fetch": 0.7,
}

_TOTAL_LAYER_COUNT = 5.0
_LAYER_ORDER: tuple[str, ...] = (
    "kpi",
    "forecast",
    "competitive",
    "cohort",
    "segmentation",
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed != parsed:
        return default
    return float(parsed)


def _log_saturating(value: int) -> float:
    n = max(1, int(value))
    numerator = log(float(n))
    denominator = log(float(n + 1))
    if denominator <= 0.0:
        return 0.0
    return _clamp01(numerator / denominator)


@dataclass(frozen=True)
class LayerIntegrity:
    available: bool
    coverage_ratio: float
    source_reliability: float
    completeness: float
    score: float
    meta: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": bool(self.available),
            "coverage_ratio": _clamp01(self.coverage_ratio),
            "source_reliability": _clamp01(self.source_reliability),
            "completeness": _clamp01(self.completeness),
            "score": _clamp01(self.score),
            "meta": dict(self.meta),
        }


class UnifiedSignalIntegrity:
    """Deterministic integrity scoring model over pipeline layers."""

    @classmethod
    def compute(cls, state: AgentState) -> dict[str, Any]:
        kpi_layer = cls._kpi_layer(state)
        forecast_layer = cls._forecast_layer(state, kpi_layer)
        competitive_layer = cls._competitive_layer(state)
        cohort_layer = cls._cohort_layer(state)
        segmentation_layer = cls._segmentation_layer(state)

        layers = {
            "kpi": kpi_layer,
            "forecast": forecast_layer,
            "competitive": competitive_layer,
            "cohort": cohort_layer,
            "segmentation": segmentation_layer,
        }

        available_layers = [name for name, layer in layers.items() if layer.available]
        # Only include layers that contributed meaningful data in the mean.
        # Layers with available=True but score=0 (insufficient_data / skipped)
        # are "not applicable" for this dataset and must not drag down
        # the overall confidence.  They are still tracked as available for
        # diagnostic visibility.
        #
        # Threshold is 0.1 (not 0.0) because layers with tiny scores
        # (e.g. forecast with r²=0.003) represent broken/degraded data —
        # including them in the mean would unfairly suppress confidence
        # computed from healthy layers.
        _MIN_LAYER_SCORE_FOR_SCORING = 0.1
        scoring_layers = [
            name for name in available_layers
            if layers[name].score >= _MIN_LAYER_SCORE_FOR_SCORING
        ]
        scoring_values = [layers[name].score for name in scoring_layers]
        kpi_gate_passed = bool(kpi_layer.score >= _KPI_MIN_GATE)

        if not scoring_values:
            overall_score = 0.0
        elif not kpi_gate_passed:
            overall_score = 0.0
        else:
            overall_score = _clamp01(mean(scoring_values))

        confidence_adjustments: list[dict[str, Any]] = []
        if scoring_layers:
            layer_weight = 1.0 / float(len(scoring_layers))
            for name in scoring_layers:
                layer_score = layers[name].score
                confidence_adjustments.append(
                    {
                        "signal": name,
                        "delta": round((layer_score - 1.0) * layer_weight, 6),
                        "reason": "layer_integrity_contribution",
                    }
                )
        if not kpi_gate_passed:
            confidence_adjustments.append(
                {
                    "signal": "kpi",
                    "delta": -1.0,
                    "reason": "kpi_gate_failed",
                }
            )

        return {
            "overall_score": round(overall_score, 6),
            "kpi_gate_passed": kpi_gate_passed,
            "layers": {name: layer.as_dict() for name, layer in layers.items()},
            "missing_layers": [name for name in _LAYER_ORDER if not layers[name].available],
            "available_layers": available_layers,
            "scoring_layers": scoring_layers,
            "confidence_adjustments": confidence_adjustments,
        }

    @classmethod
    def score_vector_from_integrity(cls, integrity: Mapping[str, Any]) -> dict[str, float]:
        """Return a flat score vector used by logs and diagnostics payloads."""
        layers = integrity.get("layers")
        if not isinstance(layers, Mapping):
            layers = {}

        def _layer_score(name: str) -> float:
            layer = layers.get(name)
            if not isinstance(layer, Mapping):
                return 0.0
            return _clamp01(_safe_float(layer.get("score"), default=0.0) or 0.0)

        overall = _clamp01(_safe_float(integrity.get("overall_score"), default=0.0) or 0.0)
        return {
            "KPI_score": round(_layer_score("kpi"), 6),
            "Forecast_score": round(_layer_score("forecast"), 6),
            "Competitive_score": round(_layer_score("competitive"), 6),
            "Cohort_score": round(_layer_score("cohort"), 6),
            "Segmentation_score": round(_layer_score("segmentation"), 6),
            "Unified_integrity_score": round(overall, 6),
        }

    @classmethod
    def score_vector(cls, state: AgentState) -> dict[str, float]:
        """Compute integrity and return the flattened score vector."""
        return cls.score_vector_from_integrity(cls.compute(state))

    @classmethod
    def _resolve_kpi_payload(cls, state: AgentState) -> tuple[dict[str, Any], str]:
        business_type = str(state.get("business_type") or "").strip().lower()
        preferred_key = KPI_KEY_BY_BUSINESS_TYPE.get(business_type)
        if preferred_key:
            candidate = state.get(preferred_key)
            payload = payload_of(candidate)
            if status_of(candidate) == "success" and isinstance(payload, dict):
                return payload, preferred_key

        for key in ("kpi_data", "saas_kpi_data", "ecommerce_kpi_data", "agency_kpi_data"):
            candidate = state.get(key)
            payload = payload_of(candidate)
            if status_of(candidate) == "success" and isinstance(payload, dict):
                return payload, key
        return {}, ""

    @classmethod
    def _kpi_layer(cls, state: AgentState) -> LayerIntegrity:
        payload, source_key = cls._resolve_kpi_payload(state)
        has_kpi_state = bool(source_key)
        if not payload:
            return LayerIntegrity(
                available=has_kpi_state,
                coverage_ratio=0.0,
                source_reliability=0.0,
                completeness=0.0,
                score=0.0,
                meta={
                    "source_key": source_key,
                    "expected_metrics": 0,
                    "available_metrics": 0,
                    "time_series_depth": 0,
                },
            )

        records = payload.get("records")
        if not isinstance(records, list):
            records = []
        compact_series = payload.get("metric_series")
        if not isinstance(compact_series, Mapping):
            compact_series = {}
        latest_computed = payload.get("latest_computed_kpis")
        if not isinstance(latest_computed, Mapping):
            latest_computed = {}
        expected = payload.get("metrics")
        if isinstance(expected, list):
            expected_metrics = [
                str(item).strip()
                for item in expected
                if str(item).strip()
            ]
        else:
            expected_metrics = []
        if not expected_metrics:
            discovered: set[str] = set(str(name).strip() for name in compact_series.keys() if str(name).strip())
            if not discovered:
                for record in records:
                    if not isinstance(record, Mapping):
                        continue
                    computed = record.get("computed_kpis")
                    if not isinstance(computed, Mapping):
                        continue
                    for metric_name in computed:
                        text = str(metric_name).strip()
                        if text:
                            discovered.add(text)
            expected_metrics = sorted(discovered)

        expected_total = len(expected_metrics)
        if expected_total == 0:
            return LayerIntegrity(
                available=True,
                coverage_ratio=0.0,
                source_reliability=0.0,
                completeness=0.0,
                score=0.0,
                meta={
                    "source_key": source_key,
                    "expected_metrics": 0,
                    "available_metrics": 0,
                    "time_series_depth": 0,
                },
            )

        weight_by_metric = {name: 0.0 for name in expected_metrics}
        depth_by_metric = {name: 0 for name in expected_metrics}

        if compact_series:
            for metric_name in expected_metrics:
                series_values = compact_series.get(metric_name)
                if isinstance(series_values, list):
                    depth = sum(1 for value in series_values if _safe_float(value, default=None) is not None)
                else:
                    depth = 0
                if depth <= 0:
                    continue
                depth_by_metric[metric_name] = depth
                latest_entry = latest_computed.get(metric_name)
                if isinstance(latest_entry, Mapping):
                    source = str(latest_entry.get("source") or "formula").strip().lower()
                else:
                    source = "formula"
                if source == "precomputed_backfill":
                    weight_by_metric[metric_name] = max(
                        weight_by_metric[metric_name],
                        _KPI_BACKFILL_WEIGHT,
                    )
                else:
                    weight_by_metric[metric_name] = max(
                        weight_by_metric[metric_name],
                        _KPI_FORMULA_WEIGHT,
                    )
        else:
            for record in records:
                if not isinstance(record, Mapping):
                    continue
                computed = record.get("computed_kpis")
                if not isinstance(computed, Mapping):
                    continue
                for metric_name in expected_metrics:
                    entry = computed.get(metric_name)
                    if entry is None:
                        continue
                    if isinstance(entry, Mapping):
                        if entry.get("error") is not None:
                            continue
                        if entry.get("value") is None:
                            continue
                        source = str(entry.get("source") or "formula").strip().lower()
                    else:
                        source = "formula"
                    depth_by_metric[metric_name] += 1
                    if source == "precomputed_backfill":
                        weight_by_metric[metric_name] = max(
                            weight_by_metric[metric_name],
                            _KPI_BACKFILL_WEIGHT,
                        )
                    else:
                        weight_by_metric[metric_name] = max(
                            weight_by_metric[metric_name],
                            _KPI_FORMULA_WEIGHT,
                        )

        available_count = sum(1 for value in weight_by_metric.values() if value > 0.0)
        weighted_sum = sum(weight_by_metric.values())
        max_depth = max(depth_by_metric.values()) if depth_by_metric else 0

        coverage_ratio = available_count / float(expected_total)
        source_reliability = (
            (weighted_sum / float(max(1, available_count)))
            if available_count > 0
            else 0.0
        )
        completeness = _clamp01(max_depth / float(_KPI_MIN_DEPTH_POINTS))
        score = _clamp01(coverage_ratio * source_reliability * completeness)

        return LayerIntegrity(
            available=True,
            coverage_ratio=coverage_ratio,
            source_reliability=source_reliability,
            completeness=completeness,
            score=score,
            meta={
                "source_key": source_key,
                "expected_metrics": expected_total,
                "available_metrics": available_count,
                "time_series_depth": max_depth,
            },
        )

    @classmethod
    def _forecast_layer(cls, state: AgentState, kpi_layer: LayerIntegrity) -> LayerIntegrity:
        envelope = state.get("forecast_data")
        if envelope is None:
            return LayerIntegrity(
                available=False,
                coverage_ratio=0.0,
                source_reliability=0.0,
                completeness=0.0,
                score=0.0,
                meta={"status": "missing"},
            )

        envelope_status = status_of(envelope)
        payload = payload_of(envelope) or {}
        if envelope_status in {"failed", "insufficient_data", "skipped"}:
            return LayerIntegrity(
                available=True,
                coverage_ratio=0.0,
                source_reliability=0.0,
                completeness=0.0,
                score=0.0,
                meta={"status": envelope_status},
            )

        forecasts = payload.get("forecasts")
        if not isinstance(forecasts, Mapping):
            forecasts = {}

        input_points: list[int] = []
        r2_values: list[float] = []
        horizons: list[int] = []
        statuses: list[str] = []
        confidence_scores: list[float] = []
        for row in forecasts.values():
            if not isinstance(row, Mapping):
                continue
            data = row.get("forecast_data")
            if not isinstance(data, Mapping):
                continue
            statuses.append(str(data.get("status") or "ok").strip().lower())
            for key in ("input_points", "historical_points", "source_points", "n_points"):
                value = _safe_int(data.get(key), default=0)
                if value > 0:
                    input_points.append(value)
                    break
            # Read R² from new regression sub-dict or legacy flat keys
            regression = data.get("regression")
            r2_found = False
            if isinstance(regression, Mapping):
                raw_r2 = _safe_float(regression.get("r_squared"))
                if raw_r2 is not None:
                    r2_values.append(_clamp01(raw_r2))
                    r2_found = True
            if not r2_found:
                for key in ("regression_r2", "r2"):
                    raw_r2 = _safe_float(data.get(key))
                    if raw_r2 is not None:
                        r2_values.append(_clamp01(raw_r2))
                        break
            # Read confidence_score from new format
            raw_conf = _safe_float(data.get("confidence_score"))
            if raw_conf is not None:
                confidence_scores.append(_clamp01(raw_conf))
            horizon = _safe_int(data.get("horizon_months"), default=0)
            if horizon <= 0:
                forecast_values = data.get("forecast")
                if isinstance(forecast_values, Mapping):
                    horizon = len(
                        [
                            key for key in forecast_values
                            if str(key).startswith("month_")
                        ]
                    )
            if horizon > 0:
                horizons.append(horizon)

        if statuses and all(status == "insufficient_data" for status in statuses):
            return LayerIntegrity(
                available=True,
                coverage_ratio=0.0,
                source_reliability=0.0,
                completeness=0.0,
                score=0.0,
                meta={"status": "insufficient_data"},
            )

        points = max(input_points) if input_points else _safe_int(
            kpi_layer.meta.get("time_series_depth"),
            default=0,
        )
        if points < _FORECAST_MIN_POINTS:
            return LayerIntegrity(
                available=True,
                coverage_ratio=0.0,
                source_reliability=0.0,
                completeness=0.0,
                score=0.0,
                meta={
                    "status": "insufficient_data",
                    "input_points": points,
                },
            )

        coverage_ratio = _log_saturating(points)
        source_reliability = (
            _clamp01(mean(r2_values))
            if r2_values
            else 1.0
        )
        horizon = max(horizons) if horizons else 3
        completeness = _clamp01(1.0 / (1.0 + log(1.0 + float(max(0, horizon)))))
        # Prefer model-provided confidence when available
        if confidence_scores:
            score = _clamp01(mean(confidence_scores))
        else:
            score = _clamp01(coverage_ratio * source_reliability * completeness)

        return LayerIntegrity(
            available=True,
            coverage_ratio=coverage_ratio,
            source_reliability=source_reliability,
            completeness=completeness,
            score=score,
            meta={
                "status": "success",
                "input_points": points,
                "horizon_months": horizon,
                "regression_r2": round(source_reliability, 6) if r2_values else None,
            },
        )

    @classmethod
    def _competitive_layer(cls, state: AgentState) -> LayerIntegrity:
        raw = state.get("competitive_context")
        if not isinstance(raw, Mapping):
            return LayerIntegrity(
                available=False,
                coverage_ratio=0.0,
                source_reliability=0.0,
                completeness=0.0,
                score=0.0,
                meta={"status": "missing"},
            )

        peer_count = _safe_int(raw.get("peer_count"), default=0)
        metrics = raw.get("metrics")
        metric_count = 0
        if isinstance(metrics, list):
            metric_count = len([m for m in metrics if str(m).strip()])
        if metric_count <= 0:
            numeric_signals = raw.get("numeric_signals")
            if isinstance(numeric_signals, list):
                names = {
                    str(item.get("metric_name") or "").strip()
                    for item in numeric_signals
                    if isinstance(item, Mapping)
                }
                metric_count = len([name for name in names if name])

        source = str(raw.get("source") or "").strip().lower()
        source_reliability = _clamp01(_COMPETITIVE_SOURCE_RELIABILITY.get(source, 0.0))
        benchmark_rows_count = max(0, _safe_int(raw.get("benchmark_rows_count"), default=0))
        completeness = 0.0
        if benchmark_rows_count > 0:
            completeness = _clamp01(log(1.0 + benchmark_rows_count) / log(2.0 + benchmark_rows_count))

        if peer_count < _COMPETITIVE_MIN_PEERS or metric_count < _COMPETITIVE_MIN_METRICS:
            return LayerIntegrity(
                available=True,
                coverage_ratio=0.0,
                source_reliability=source_reliability,
                completeness=completeness,
                score=0.0,
                meta={
                    "status": "insufficient_data",
                    "peer_count": peer_count,
                    "metric_count": metric_count,
                    "source": source,
                    "benchmark_rows_count": benchmark_rows_count,
                },
            )

        peer_term = _log_saturating(peer_count)
        metric_term = _log_saturating(metric_count)
        coverage_ratio = _clamp01(sqrt(peer_term * metric_term))
        score = _clamp01(coverage_ratio * source_reliability * completeness)

        return LayerIntegrity(
            available=True,
            coverage_ratio=coverage_ratio,
            source_reliability=source_reliability,
            completeness=completeness,
            score=score,
            meta={
                "status": "success",
                "peer_count": peer_count,
                "metric_count": metric_count,
                "source": source,
                "benchmark_rows_count": benchmark_rows_count,
            },
        )

    @classmethod
    def _cohort_payload(cls, state: AgentState) -> dict[str, Any]:
        segmentation = payload_of(state.get("segmentation")) or {}
        if isinstance(segmentation, Mapping):
            candidate = segmentation.get("cohort_analytics")
            if isinstance(candidate, Mapping):
                return dict(candidate)
        candidate = payload_of(state.get("cohort_data")) or {}
        return dict(candidate) if isinstance(candidate, Mapping) else {}

    @classmethod
    def _cohort_layer(cls, state: AgentState) -> LayerIntegrity:
        payload = cls._cohort_payload(state)
        if not payload:
            has_envelope = state.get("cohort_data") is not None or state.get("segmentation") is not None
            return LayerIntegrity(
                available=has_envelope,
                coverage_ratio=0.0,
                source_reliability=0.0 if not has_envelope else 1.0,
                completeness=0.0,
                score=0.0,
                meta={"status": "missing"},
            )

        keys_present = False
        cohort_count = 0
        cohort_keys = payload.get("cohort_keys")
        if isinstance(cohort_keys, list) and any(str(item).strip() for item in cohort_keys):
            keys_present = True
        cohorts_by_key = payload.get("cohorts_by_key")
        if isinstance(cohorts_by_key, Mapping):
            keys_present = True
            for value in cohorts_by_key.values():
                if isinstance(value, Mapping):
                    cohort_count += max(0, _safe_int(value.get("count"), default=0))

        if not keys_present:
            return LayerIntegrity(
                available=True,
                coverage_ratio=0.0,
                source_reliability=1.0,
                completeness=0.0,
                score=0.0,
                meta={"status": "missing_keys", "cohort_count": 0},
            )

        if cohort_count >= 2:
            completeness = 1.0
        elif cohort_count == 1:
            completeness = 0.3
        else:
            completeness = 0.0

        coverage_ratio = 1.0 if keys_present else 0.0
        source_reliability = 1.0
        score = _clamp01(coverage_ratio * source_reliability * completeness)

        return LayerIntegrity(
            available=True,
            coverage_ratio=coverage_ratio,
            source_reliability=source_reliability,
            completeness=completeness,
            score=score,
            meta={"status": "success" if score > 0 else "insufficient_data", "cohort_count": cohort_count},
        )

    @classmethod
    def _segmentation_payload(cls, state: AgentState) -> tuple[dict[str, Any], bool]:
        envelope = state.get("segmentation")
        payload = payload_of(envelope) if envelope is not None else None
        if isinstance(payload, Mapping):
            return dict(payload), True
        return {}, envelope is not None

    @classmethod
    def _segmentation_layer(cls, state: AgentState) -> LayerIntegrity:
        payload, has_envelope = cls._segmentation_payload(state)
        if not payload:
            return LayerIntegrity(
                available=has_envelope,
                coverage_ratio=0.0,
                source_reliability=0.0 if not has_envelope else 1.0,
                completeness=0.0,
                score=0.0,
                meta={"status": "missing"},
            )

        segment_count = 0
        keys_present = False

        top_contributors = payload.get("top_contributors")
        if isinstance(top_contributors, list):
            keys_present = True
            segment_count = max(segment_count, len(top_contributors))

        by_dimension = payload.get("by_dimension")
        if isinstance(by_dimension, Mapping):
            keys_present = True
            discovered: set[str] = set()
            for value in by_dimension.values():
                if not isinstance(value, Mapping):
                    continue
                contributors = value.get("contributors")
                if not isinstance(contributors, list):
                    continue
                for item in contributors:
                    if isinstance(item, Mapping):
                        name = str(item.get("name") or "").strip()
                        if name:
                            discovered.add(name)
            segment_count = max(segment_count, len(discovered))

        if not keys_present:
            return LayerIntegrity(
                available=True,
                coverage_ratio=0.0,
                source_reliability=1.0,
                completeness=0.0,
                score=0.0,
                meta={"status": "missing_keys", "segment_count": 0},
            )

        if segment_count >= 2:
            completeness = 1.0
        elif segment_count == 1:
            completeness = 0.4
        else:
            completeness = 0.0

        coverage_ratio = 1.0 if keys_present else 0.0
        source_reliability = 1.0
        score = _clamp01(coverage_ratio * source_reliability * completeness)

        return LayerIntegrity(
            available=True,
            coverage_ratio=coverage_ratio,
            source_reliability=source_reliability,
            completeness=completeness,
            score=score,
            meta={"status": "success" if score > 0 else "insufficient_data", "segment_count": segment_count},
        )

    @staticmethod
    def metric_confidence(
        values: list[float],
        *,
        signals: dict[str, float | None] | None = None,
        tier_cap: float | None = None,
    ) -> dict[str, Any]:
        """Compute unified confidence for a single metric series.

        Delegates to the composable confidence scoring model which evaluates
        four dimensions: depth, volatility, anomaly presence, and signal
        consistency.  This provides a single entry-point for nodes that need
        confidence without constructing the full layer integrity model.

        Parameters
        ----------
        values:
            Time-series data points, oldest first.
        signals:
            Optional directional signals for consistency scoring.
        tier_cap:
            Optional hard ceiling (e.g. 0.40 for minimal-tier data).

        Returns
        -------
        dict
            Full confidence result from
            ``app.services.statistics.confidence_scoring.compute_confidence``.
        """
        from app.services.statistics.confidence_scoring import compute_confidence

        return compute_confidence(values, signals=signals, tier_cap=tier_cap)
