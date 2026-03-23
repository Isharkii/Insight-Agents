"""Unified deterministic signal integrity model.

This module centralizes confidence math across pipeline nodes.  It replaces
ad-hoc penalty deltas with deterministic layer scores derived from state.

Architecture
============

Signal Authority Hierarchy (highest → lowest):
    KPI > Cohort > Segmentation > Unit Economics > Forecast > Scenario

Gating Modes:
    HARD BLOCK  — pipeline failure or zero usable KPI data
    SOFT DEGRADE — insufficient_data or low-confidence signals
    ISOLATE     — broken modules (e.g. forecast R² < 0.1)

Layers with ``gate_status == "isolate"`` are excluded from scoring AND
confidence propagation.  Degraded layers are included at reduced weight.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from math import log, sqrt
from statistics import mean
from typing import Any, Mapping

from agent.graph_config import (
    KPI_KEY_BY_BUSINESS_TYPE,
    graph_node_config_for_business_type,
    signal_name_for_state_key,
)
from agent.nodes.node_result import payload_of, status_of
from agent.state import AgentState

logger = logging.getLogger(__name__)

# ── Thresholds ──────────────────────────────────────────────────────────────

_KPI_MIN_GATE = 0.3
_KPI_FORMULA_WEIGHT = 1.0
_KPI_BACKFILL_WEIGHT = 0.7
_KPI_MIN_DEPTH_POINTS = 3

_FORECAST_MIN_POINTS = 6
_FORECAST_MIN_R2 = 0.10          # R² below this → forecast isolated (no predictive power)
_FORECAST_LOW_R2 = 0.30          # R² below this → forecast degraded (weak fit)

_COMPETITIVE_MIN_PEERS = 2
_COMPETITIVE_MIN_METRICS = 2
_COMPETITIVE_SOURCE_RELIABILITY = {
    "local_db": 1.0,
    "deterministic_local": 1.0,
    "external_fetch": 0.7,
}
# Default reliability for unrecognised competitive sources (prevents silent
# zeroing when a new source type is introduced).
_COMPETITIVE_DEFAULT_SOURCE_RELIABILITY = 0.5

# ── Signal Authority Hierarchy ──────────────────────────────────────────────
# Higher-authority signals carry more weight in overall scoring.  Lower
# signals cannot inflate confidence beyond what higher signals support.

_SIGNAL_AUTHORITY: dict[str, float] = {
    "kpi":           1.0,      # ground truth – highest authority
    "cohort":        0.8,
    "segmentation":  0.6,
    "competitive":   0.5,
    "forecast":      0.4,      # model-derived – lowest authority
}

_TOTAL_LAYER_COUNT = 5.0
_LAYER_ORDER: tuple[str, ...] = (
    "kpi",
    "forecast",
    "competitive",
    "cohort",
    "segmentation",
)


# ── Gating Modes ────────────────────────────────────────────────────────────

class GateStatus(str, Enum):
    """Signal gating classification."""
    PASS = "pass"
    DEGRADE = "degrade"
    ISOLATE = "isolate"
    BLOCK = "block"


def gate_signal(status: str, confidence: float) -> GateStatus:
    """Classify a signal into a gating mode.

    BLOCK   — failed / explicitly blocked
    ISOLATE — available but confidence too low to trust (< 0.2)
    DEGRADE — insufficient_data or low confidence
    PASS    — healthy signal
    """
    normalized = str(status or "").strip().lower()
    if normalized in ("failed", "blocked"):
        return GateStatus.BLOCK
    if confidence < 0.2 and normalized != "success":
        return GateStatus.ISOLATE
    if normalized == "insufficient_data":
        return GateStatus.DEGRADE
    if confidence < 0.3:
        return GateStatus.DEGRADE
    return GateStatus.PASS


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
    gate_status: GateStatus = GateStatus.PASS

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": bool(self.available),
            "coverage_ratio": _clamp01(self.coverage_ratio),
            "source_reliability": _clamp01(self.source_reliability),
            "completeness": _clamp01(self.completeness),
            "score": _clamp01(self.score),
            "gate_status": self.gate_status.value,
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

        # ── Gating classification ───────────────────────────────────
        # ISOLATE: broken modules excluded from scoring + propagation
        # DEGRADE: weak modules included at reduced authority weight
        # PASS:    healthy modules at full authority weight
        isolated_layers: list[str] = []
        degraded_layers: list[str] = []
        scoring_layers: list[str] = []
        reasoning_warnings: list[str] = []

        for name in available_layers:
            layer = layers[name]
            gs = layer.gate_status
            if gs == GateStatus.BLOCK or gs == GateStatus.ISOLATE:
                isolated_layers.append(name)
                reasoning_warnings.append(
                    f"{name} layer isolated: {layer.meta.get('status', 'unknown')} "
                    f"(score={layer.score:.3f}, gate={gs.value})"
                )
            elif gs == GateStatus.DEGRADE:
                degraded_layers.append(name)
                # Only include degraded layers in scoring if they have a
                # non-trivial score — a degraded layer at score=0 has nothing
                # to contribute and would only dilute healthy layers.
                if layer.score >= 0.01:
                    scoring_layers.append(name)
                reasoning_warnings.append(
                    f"{name} layer degraded: {layer.meta.get('status', 'unknown')} "
                    f"(score={layer.score:.3f})"
                )
            elif layer.score >= 0.01:
                scoring_layers.append(name)

        # ── Authority-weighted scoring ──────────────────────────────
        # Each layer contributes score × authority_weight.  Degraded layers
        # have their authority halved.  Isolated layers contribute nothing.
        kpi_gate_passed = bool(kpi_layer.score >= _KPI_MIN_GATE)

        if not scoring_layers:
            overall_score = 0.0
        else:
            weighted_sum = 0.0
            weight_sum = 0.0
            for name in scoring_layers:
                authority = _SIGNAL_AUTHORITY.get(name, 0.3)
                if name in degraded_layers:
                    authority *= 0.5  # halve authority for degraded
                layer_score = layers[name].score
                weighted_sum += layer_score * authority
                weight_sum += authority
            raw_score = weighted_sum / weight_sum if weight_sum > 0 else 0.0

            # ── KPI continuous penalty (replaces binary cliff) ──────
            # Instead of zeroing overall_score when KPI < 0.3, apply a
            # smooth nonlinear penalty that preserves other-layer signal.
            #
            # penalty = (kpi_score / threshold)^2  when kpi_score < threshold
            #   → score 0.29 → penalty ≈ 0.93  (near miss, mild)
            #   → score 0.15 → penalty ≈ 0.25  (severe, but not zero)
            #   → score 0.00 → penalty = 0.00  (total KPI failure)
            if not kpi_gate_passed and kpi_layer.score > 0:
                kpi_penalty = (kpi_layer.score / _KPI_MIN_GATE) ** 2
                raw_score *= kpi_penalty
                reasoning_warnings.append(
                    f"KPI below gate ({kpi_layer.score:.3f} < {_KPI_MIN_GATE}): "
                    f"applied continuous penalty factor {kpi_penalty:.3f}"
                )
            elif kpi_layer.score == 0 and kpi_layer.available:
                raw_score = 0.0
                reasoning_warnings.append(
                    "KPI layer has zero score — overall confidence forced to 0.0"
                )

            overall_score = _clamp01(raw_score)

        # ── Confidence caps (monotonic degradation) ──────────────────
        # More missing/degraded signals → strictly lower confidence.
        # Caps are applied sequentially; the most restrictive wins.
        base_score = overall_score
        caps_applied: list[dict[str, Any]] = []

        valid_layer_count = len(scoring_layers)
        kpi_depth = _safe_int(kpi_layer.meta.get("time_series_depth"), default=0)
        kpi_coverage = kpi_layer.coverage_ratio
        forecast_usable = forecast_layer.gate_status in (
            GateStatus.PASS, GateStatus.DEGRADE,
        )

        # Cap: critical layer (KPI) degraded → ≤ 0.35
        if kpi_layer.gate_status == GateStatus.DEGRADE:
            cap = 0.35
            if overall_score > cap:
                caps_applied.append({
                    "cap": "kpi_degraded", "limit": cap,
                })
                overall_score = cap
                reasoning_warnings.append(
                    f"KPI layer degraded: confidence capped at {cap}"
                )

        # Cap: forecast unusable (isolated/blocked) → ≤ 0.30
        if not forecast_usable:
            cap = 0.30
            if overall_score > cap:
                caps_applied.append({
                    "cap": "forecast_unusable", "limit": cap,
                    "gate_status": forecast_layer.gate_status.value,
                })
                overall_score = cap
                reasoning_warnings.append(
                    f"Forecast unusable ({forecast_layer.gate_status.value}): "
                    f"confidence capped at {cap}"
                )

        # Cap: insufficient analytical layers (< 3) → ≤ 0.25
        if valid_layer_count < 3:
            cap = 0.25
            if overall_score > cap:
                caps_applied.append({
                    "cap": "insufficient_layers", "limit": cap,
                    "valid_layers": valid_layer_count,
                })
                overall_score = cap
                reasoning_warnings.append(
                    f"Only {valid_layer_count} valid layer(s): "
                    f"confidence capped at {cap}"
                )

        overall_score = _clamp01(overall_score)

        # ── Insight quality classification ────────────────────────────
        # Determines output state BEFORE synthesis:
        #   full_insight    — 3+ valid layers, KPI gate passed
        #   partial_insight — 1-2 valid layers (recommendations stripped)
        #   blocked         — 0 valid layers or KPI hard-failed
        if valid_layer_count == 0 or (kpi_layer.available and kpi_layer.score == 0):
            insight_quality = "blocked"
        elif valid_layer_count < 3:
            insight_quality = "partial_insight"
        else:
            insight_quality = "full_insight"

        # ── Layer classification vector ───────────────────────────────
        layer_classification: dict[str, str] = {}
        for name in _LAYER_ORDER:
            layer = layers[name]
            if not layer.available:
                layer_classification[name] = "missing"
            elif name in isolated_layers:
                layer_classification[name] = "isolated"
            elif name in degraded_layers:
                layer_classification[name] = "degraded"
            elif name in scoring_layers:
                layer_classification[name] = "valid"
            else:
                layer_classification[name] = "inactive"

        # ── Confidence adjustments for diagnostics ──────────────────
        confidence_adjustments: list[dict[str, Any]] = []
        if scoring_layers:
            for name in scoring_layers:
                authority = _SIGNAL_AUTHORITY.get(name, 0.3)
                if name in degraded_layers:
                    authority *= 0.5
                layer_score = layers[name].score
                confidence_adjustments.append(
                    {
                        "signal": name,
                        "authority_weight": round(authority, 3),
                        "layer_score": round(layer_score, 6),
                        "delta": round((layer_score - 1.0) * authority, 6),
                        "reason": (
                            "degraded_contribution" if name in degraded_layers
                            else "layer_integrity_contribution"
                        ),
                    }
                )
        for name in isolated_layers:
            confidence_adjustments.append(
                {
                    "signal": name,
                    "authority_weight": 0.0,
                    "layer_score": round(layers[name].score, 6),
                    "delta": 0.0,
                    "reason": f"isolated_{layers[name].gate_status.value}",
                }
            )
        if not kpi_gate_passed:
            confidence_adjustments.append(
                {
                    "signal": "kpi",
                    "delta": round(-(1.0 - (kpi_layer.score / _KPI_MIN_GATE) ** 2), 6) if kpi_layer.score > 0 else -1.0,
                    "reason": "kpi_gate_continuous_penalty",
                }
            )

        # ── KPI coverage validation ──────────────────────────────────
        kpi_coverage_report = cls.validate_kpi_coverage(state)

        # ── Missing signal detection ─────────────────────────────────
        missing_signal_report = cls._detect_missing_signals(state, layers)
        missing_required_count = sum(
            1 for s in missing_signal_report["missing_signals"]
            if s["classification"] == "required"
        )

        # Confidence penalty for missing required signals:
        # Each missing required signal reduces confidence by 15% multiplicatively
        # (max 3 → factor 0.614).  Multiplicative avoids zeroing already-low scores.
        if missing_required_count > 0:
            factor = max(0.50, 1.0 - 0.15 * missing_required_count)
            pre_penalty = overall_score
            overall_score = _clamp01(overall_score * factor)
            caps_applied.append({
                "cap": "missing_required_signals",
                "limit": round(overall_score, 6),
                "missing_count": missing_required_count,
                "factor": round(factor, 6),
            })
            reasoning_warnings.append(
                f"{missing_required_count} missing required signal(s): "
                f"confidence reduced by factor {factor:.2f} "
                f"({pre_penalty:.3f} → {overall_score:.3f})"
            )

        return {
            "overall_score": round(overall_score, 6),
            "kpi_gate_passed": kpi_gate_passed,
            "kpi_depth": kpi_depth,
            "kpi_coverage_ratio": round(kpi_coverage, 6),
            "valid_layer_count": valid_layer_count,
            "forecast_usable": forecast_usable,
            "insight_quality": insight_quality,
            "layer_classification": layer_classification,
            "missing_signal_report": missing_signal_report,
            "kpi_coverage_report": kpi_coverage_report,
            "confidence_breakdown": {
                "base": round(base_score, 6),
                "penalties": {
                    "kpi_penalty_applied": not kpi_gate_passed,
                    "caps_applied": caps_applied,
                },
                "final": round(overall_score, 6),
            },
            "layers": {name: layer.as_dict() for name, layer in layers.items()},
            "missing_layers": [name for name in _LAYER_ORDER if not layers[name].available],
            "available_layers": available_layers,
            "scoring_layers": scoring_layers,
            "isolated_layers": isolated_layers,
            "degraded_layers": degraded_layers,
            "reasoning_warnings": reasoning_warnings,
            "confidence_adjustments": confidence_adjustments,
        }

    # ── Confidence cap constants ──────────────────────────────────
    _CAP_KPI_DEGRADED = 0.35
    _CAP_FORECAST_UNUSABLE = 0.30
    _CAP_INSUFFICIENT_LAYERS = 0.25

    @classmethod
    def enforce_final_confidence_caps(
        cls,
        confidence: float,
        integrity: dict[str, Any],
    ) -> dict[str, Any]:
        """FINAL confidence enforcement — no downstream module may modify after.

        Re-applies all caps as a safety net and returns a spec-format trace::

            {
                "raw_confidence": float,
                "applied_caps": [...],
                "final_confidence": float,
            }
        """
        raw = float(confidence)
        capped = raw
        applied_caps: list[dict[str, Any]] = []

        layer_classification = integrity.get("layer_classification", {})
        valid_layer_count = int(integrity.get("valid_layer_count", 0))
        forecast_usable = bool(integrity.get("forecast_usable", True))

        kpi_status = layer_classification.get("kpi", "missing")

        # Cap 1: KPI degraded → ≤ 0.35
        if kpi_status == "degraded" and capped > cls._CAP_KPI_DEGRADED:
            applied_caps.append({
                "cap": "kpi_degraded",
                "limit": cls._CAP_KPI_DEGRADED,
                "before": round(capped, 6),
            })
            capped = cls._CAP_KPI_DEGRADED

        # Cap 2: forecast unusable → ≤ 0.30
        if not forecast_usable and capped > cls._CAP_FORECAST_UNUSABLE:
            applied_caps.append({
                "cap": "forecast_unusable",
                "limit": cls._CAP_FORECAST_UNUSABLE,
                "before": round(capped, 6),
            })
            capped = cls._CAP_FORECAST_UNUSABLE

        # Cap 3: insufficient layers (< 3) → ≤ 0.25
        if valid_layer_count < 3 and capped > cls._CAP_INSUFFICIENT_LAYERS:
            applied_caps.append({
                "cap": "insufficient_layers",
                "limit": cls._CAP_INSUFFICIENT_LAYERS,
                "valid_layers": valid_layer_count,
                "before": round(capped, 6),
            })
            capped = cls._CAP_INSUFFICIENT_LAYERS

        capped = max(0.0, min(1.0, capped))

        return {
            "raw_confidence": round(raw, 6),
            "applied_caps": applied_caps,
            "final_confidence": round(capped, 6),
        }

    # ── KPI coverage validation ──────────────────────────────────
    _MIN_KPI_COVERAGE_RATIO = 0.5

    @classmethod
    def validate_kpi_coverage(
        cls,
        state: AgentState,
    ) -> dict[str, Any]:
        """Validate KPI metric coverage against expected metrics.

        Returns structured report::

            {
                "coverage_ratio": float,
                "expected_metrics": [...],
                "available_metrics": [...],
                "missing_metrics": [...],
                "sufficient": bool,
                "impact": str | None,
            }

        When ``coverage_ratio < 0.5``, synthesis MUST be blocked.
        """
        kpi_layer = cls._kpi_layer(state)
        meta = kpi_layer.meta

        expected_count = int(meta.get("expected_metrics", 0))
        available_count = int(meta.get("available_metrics", 0))
        coverage_ratio = kpi_layer.coverage_ratio

        # Resolve expected/available metric names for the report
        payload, _ = cls._resolve_kpi_payload(state)
        expected_names: list[str] = []
        available_names: list[str] = []

        if payload:
            expected_raw = payload.get("metrics")
            compact_series = payload.get("metric_series")
            records = payload.get("records")

            # Build expected list
            if isinstance(expected_raw, list):
                expected_names = [str(m).strip() for m in expected_raw if str(m).strip()]
            elif isinstance(compact_series, Mapping):
                expected_names = sorted(str(k).strip() for k in compact_series if str(k).strip())
            elif isinstance(records, list):
                discovered: set[str] = set()
                for record in records:
                    if isinstance(record, Mapping):
                        computed = record.get("computed_kpis")
                        if isinstance(computed, Mapping):
                            for name in computed:
                                text = str(name).strip()
                                if text:
                                    discovered.add(text)
                expected_names = sorted(discovered)

            # Build available list (metrics with actual data)
            if isinstance(compact_series, Mapping):
                for name in expected_names:
                    series = compact_series.get(name)
                    if isinstance(series, list) and any(
                        _safe_float(v, default=None) is not None for v in series
                    ):
                        available_names.append(name)
            elif isinstance(records, list):
                present: set[str] = set()
                for record in records:
                    if not isinstance(record, Mapping):
                        continue
                    computed = record.get("computed_kpis")
                    if not isinstance(computed, Mapping):
                        continue
                    for name in expected_names:
                        entry = computed.get(name)
                        if entry is not None:
                            if isinstance(entry, Mapping):
                                if entry.get("error") is None and entry.get("value") is not None:
                                    present.add(name)
                            else:
                                present.add(name)
                available_names = sorted(present)

        missing_names = [m for m in expected_names if m not in available_names]
        sufficient = coverage_ratio >= cls._MIN_KPI_COVERAGE_RATIO

        return {
            "coverage_ratio": round(coverage_ratio, 6),
            "expected_metrics": expected_names,
            "available_metrics": available_names,
            "missing_metrics": missing_names,
            "sufficient": sufficient,
            "impact": (
                "insufficient KPI reliability"
                if not sufficient
                else None
            ),
        }

    @classmethod
    def _detect_missing_signals(
        cls,
        state: AgentState,
        layers: dict[str, "LayerIntegrity"],
    ) -> dict[str, Any]:
        """Detect missing required and optional signals.

        Returns structured report::

            {
                "missing_signals": [
                    {
                        "signal": "forecast",
                        "state_key": "forecast_data",
                        "classification": "required" | "optional",
                        "status": "missing" | "failed" | "skipped",
                        "impact": "confidence penalty" | "reduced coverage",
                    },
                    ...
                ],
                "required_missing_count": int,
                "optional_missing_count": int,
                "actions_taken": [...],
            }
        """
        _HARD_FAIL = {"failed", "skipped"}

        business_type = str(state.get("business_type") or "")
        config = graph_node_config_for_business_type(business_type)

        missing_signals: list[dict[str, Any]] = []
        actions_taken: list[str] = []

        all_keys = list(dict.fromkeys((*config.required, *config.optional)))
        for key in all_keys:
            is_required = key in config.required
            classification = "required" if is_required else "optional"
            signal = signal_name_for_state_key(key)
            value = state.get(key)

            if value is None:
                missing_signals.append({
                    "signal": signal,
                    "state_key": key,
                    "classification": classification,
                    "status": "missing",
                    "impact": (
                        "confidence penalty"
                        if is_required
                        else "reduced coverage"
                    ),
                })
                if is_required:
                    actions_taken.append(
                        f"required signal '{signal}' missing: "
                        f"confidence penalised"
                    )
                else:
                    actions_taken.append(
                        f"optional signal '{signal}' missing: "
                        f"coverage reduced"
                    )
                continue

            status = status_of(value)
            if status in _HARD_FAIL:
                missing_signals.append({
                    "signal": signal,
                    "state_key": key,
                    "classification": classification,
                    "status": status,
                    "impact": (
                        "confidence penalty"
                        if is_required
                        else "reduced coverage"
                    ),
                })
                if is_required:
                    actions_taken.append(
                        f"required signal '{signal}' {status}: "
                        f"confidence penalised"
                    )

        # Also check layer-level missing (layers not backed by a state key
        # but detected as unavailable by the integrity model).
        for name, layer in layers.items():
            if not layer.available:
                # Already captured by state key check above? Skip duplicates.
                already = any(s["signal"] == name for s in missing_signals)
                if not already:
                    missing_signals.append({
                        "signal": name,
                        "state_key": None,
                        "classification": "optional",
                        "status": "missing",
                        "impact": "reduced coverage",
                    })

        required_missing = sum(
            1 for s in missing_signals if s["classification"] == "required"
        )
        optional_missing = sum(
            1 for s in missing_signals if s["classification"] == "optional"
        )

        if not missing_signals:
            actions_taken.append("all signals present")

        return {
            "missing_signals": missing_signals,
            "required_missing_count": required_missing,
            "optional_missing_count": optional_missing,
            "actions_taken": actions_taken,
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
    def filter_layers_for_downstream(
        cls,
        state: AgentState,
        *,
        allow_degraded: bool = True,
    ) -> dict[str, Any]:
        """Centralized layer filter for downstream consumers.

        Returns a dict with:
            eligible_layers: list of layer names safe for downstream use
            isolated_layers: list of layer names that MUST be excluded
            layer_classification: full classification dict
            warnings: list of filtering actions taken

        Downstream nodes (conflict detection, risk scoring, prioritization,
        recommendations) MUST use this filter instead of reading raw layer
        outputs directly.
        """
        integrity = cls.compute(state)
        classification = integrity.get("layer_classification", {})
        isolated = integrity.get("isolated_layers", [])
        degraded = integrity.get("degraded_layers", [])

        eligible: list[str] = []
        warnings: list[str] = []

        for name, status in classification.items():
            if status == "valid":
                eligible.append(name)
            elif status == "degraded" and allow_degraded:
                eligible.append(name)
                warnings.append(
                    f"{name} layer degraded: included with reduced authority"
                )
            elif status in ("isolated", "missing", "inactive"):
                warnings.append(
                    f"{name} layer {status}: excluded from downstream"
                )

        return {
            "eligible_layers": eligible,
            "isolated_layers": list(isolated),
            "degraded_layers": list(degraded),
            "layer_classification": classification,
            "warnings": warnings,
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
                gate_status=GateStatus.BLOCK,
            )

        envelope_status = status_of(envelope)
        payload = payload_of(envelope) or {}
        if envelope_status in {"failed", "skipped"}:
            return LayerIntegrity(
                available=True,
                coverage_ratio=0.0,
                source_reliability=0.0,
                completeness=0.0,
                score=0.0,
                meta={"status": envelope_status},
                gate_status=GateStatus.BLOCK,
            )
        if envelope_status == "insufficient_data":
            return LayerIntegrity(
                available=True,
                coverage_ratio=0.0,
                source_reliability=0.0,
                completeness=0.0,
                score=0.0,
                meta={"status": envelope_status},
                gate_status=GateStatus.DEGRADE,
            )

        forecasts = payload.get("forecasts")
        if not isinstance(forecasts, Mapping):
            forecasts = {}

        input_points: list[int] = []
        r2_values: list[float] = []
        horizons: list[int] = []
        statuses: list[str] = []
        confidence_scores: list[float] = []
        forecast_slopes: list[float] = []
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
                # Capture slope for direction validation
                raw_slope = _safe_float(regression.get("slope"))
                if raw_slope is not None:
                    forecast_slopes.append(raw_slope)
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
                gate_status=GateStatus.DEGRADE,
            )

        # ── Dependency check: forecast depends on KPI depth ─────────
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
                    "dependency_note": f"needs >= {_FORECAST_MIN_POINTS} points, has {points}",
                },
                gate_status=GateStatus.DEGRADE,
            )

        coverage_ratio = _log_saturating(points)
        avg_r2 = _clamp01(mean(r2_values)) if r2_values else 1.0
        source_reliability = avg_r2
        horizon = max(horizons) if horizons else 3
        completeness = _clamp01(1.0 / (1.0 + log(1.0 + float(max(0, horizon)))))

        # ── Forecast R² validation (strict) ─────────────────────────
        # R² < 0.2 → ISOLATE (unusable, zero weight)
        # R² < 0.4 → DEGRADE (included at reduced weight)
        if r2_values and avg_r2 < _FORECAST_MIN_R2:
            logger.info(
                "Forecast isolated: R²=%.4f < %.2f minimum",
                avg_r2, _FORECAST_MIN_R2,
            )
            return LayerIntegrity(
                available=True,
                coverage_ratio=coverage_ratio,
                source_reliability=source_reliability,
                completeness=completeness,
                score=_clamp01(avg_r2),  # preserve raw score for diagnostics
                meta={
                    "status": "isolated_low_r2",
                    "input_points": points,
                    "horizon_months": horizon,
                    "regression_r2": round(avg_r2, 6),
                    "isolation_reason": f"R²={avg_r2:.4f} < {_FORECAST_MIN_R2}",
                },
                gate_status=GateStatus.ISOLATE,
            )

        forecast_gate = GateStatus.PASS
        meta_status = "success"

        if r2_values and avg_r2 < _FORECAST_LOW_R2:
            forecast_gate = GateStatus.DEGRADE
            meta_status = "degraded_low_r2"
            logger.info(
                "Forecast degraded: R²=%.4f < %.2f threshold",
                avg_r2, _FORECAST_LOW_R2,
            )

        # ── Direction contradiction check ───────────────────────────
        # If forecast slope contradicts KPI trend direction AND R² is low,
        # penalize the forecast further.
        kpi_growth = state.get("growth_data")
        kpi_growth_payload = payload_of(kpi_growth) if kpi_growth else None
        if isinstance(kpi_growth_payload, Mapping) and forecast_slopes:
            kpi_short = _safe_float(kpi_growth_payload.get("short_growth"))
            avg_slope = mean(forecast_slopes)
            if kpi_short is not None and avg_slope != 0:
                kpi_direction = 1 if kpi_short > 0 else (-1 if kpi_short < 0 else 0)
                forecast_direction = 1 if avg_slope > 0 else (-1 if avg_slope < 0 else 0)
                if kpi_direction != 0 and forecast_direction != 0 and kpi_direction != forecast_direction:
                    # Contradiction: forecast says opposite of KPI
                    if avg_r2 < 0.5:
                        forecast_gate = GateStatus.ISOLATE
                        meta_status = "isolated_direction_conflict"
                        logger.info(
                            "Forecast isolated: direction contradicts KPI "
                            "(forecast_slope=%.4f, kpi_short=%.4f) with low R²=%.4f",
                            avg_slope, kpi_short, avg_r2,
                        )

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
                "status": meta_status,
                "input_points": points,
                "horizon_months": horizon,
                "regression_r2": round(avg_r2, 6) if r2_values else None,
                "forecast_slopes": [round(s, 6) for s in forecast_slopes] if forecast_slopes else None,
            },
            gate_status=forecast_gate,
        )

    @classmethod
    def _competitive_layer(cls, state: AgentState) -> LayerIntegrity:
        # ── Prefer local benchmark_data over external competitive_context ──
        # benchmark_data comes from the deterministic peer benchmark service
        # (local DB, validated ranking, composite scoring).  competitive_context
        # comes from external intelligence fetching (web scraping, APIs).
        benchmark_envelope = state.get("benchmark_data")
        benchmark_payload = payload_of(benchmark_envelope) if benchmark_envelope else None
        benchmark_status = status_of(benchmark_envelope) if benchmark_envelope else None

        if isinstance(benchmark_payload, Mapping) and benchmark_status == "success":
            return cls._competitive_layer_from_benchmark(benchmark_payload)

        # Fall back to external competitive_context
        raw = state.get("competitive_context")
        if not isinstance(raw, Mapping):
            # Check if benchmark envelope exists but is partial/failed
            if benchmark_envelope is not None:
                return LayerIntegrity(
                    available=True,
                    coverage_ratio=0.0,
                    source_reliability=1.0,
                    completeness=0.0,
                    score=0.0,
                    meta={
                        "status": benchmark_status or "missing",
                        "source": "benchmark_node",
                    },
                    gate_status=GateStatus.DEGRADE,
                )
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
        source_reliability = _clamp01(
            _COMPETITIVE_SOURCE_RELIABILITY.get(source, _COMPETITIVE_DEFAULT_SOURCE_RELIABILITY)
        )
        benchmark_rows_count = max(0, _safe_int(raw.get("benchmark_rows_count"), default=0))
        numeric_signals = raw.get("numeric_signals")
        sample_sizes: list[int] = []
        if isinstance(numeric_signals, list):
            for item in numeric_signals:
                if not isinstance(item, Mapping):
                    continue
                sample_size = _safe_int(item.get("sample_size"), default=0)
                if sample_size > 0:
                    sample_sizes.append(sample_size)

        completeness = 0.0
        if benchmark_rows_count > 0:
            completeness = _clamp01(log(1.0 + benchmark_rows_count) / log(2.0 + benchmark_rows_count))
        elif sample_sizes:
            # External benchmark sources may have no canonical benchmark rows
            # but still provide structured sample-backed numeric aggregates.
            # Treat that as degraded completeness instead of hard zero.
            total_samples = sum(sample_sizes)
            sample_term = _clamp01(
                log(1.0 + total_samples) / log(25.0 + total_samples)
            )
            signal_term = _log_saturating(len(sample_sizes) + 1)
            completeness = _clamp01(sqrt(sample_term * signal_term))

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
                    "sample_sizes": sample_sizes,
                },
                gate_status=GateStatus.DEGRADE,
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
                "sample_sizes": sample_sizes,
            },
        )

    @classmethod
    def _competitive_layer_from_benchmark(
        cls, payload: Mapping[str, Any],
    ) -> LayerIntegrity:
        """Build competitive layer integrity from local benchmark_data."""
        peer_selection = payload.get("peer_selection") or {}
        selected_peers = peer_selection.get("selected_peers") or []
        peer_count = len(selected_peers)

        composite = payload.get("composite") or {}
        ranking = payload.get("ranking") or {}
        metric_specs = payload.get("metric_comparison_specs") or {}
        metric_count = len(metric_specs)

        # Local DB benchmarks have maximum source reliability
        source_reliability = 1.0

        if peer_count < _COMPETITIVE_MIN_PEERS or metric_count < _COMPETITIVE_MIN_METRICS:
            return LayerIntegrity(
                available=True,
                coverage_ratio=0.0,
                source_reliability=source_reliability,
                completeness=0.0,
                score=0.0,
                meta={
                    "status": "insufficient_data",
                    "peer_count": peer_count,
                    "metric_count": metric_count,
                    "source": "benchmark_node",
                },
                gate_status=GateStatus.DEGRADE,
            )

        peer_term = _log_saturating(peer_count)
        metric_term = _log_saturating(metric_count)
        coverage_ratio = _clamp01(sqrt(peer_term * metric_term))

        # Completeness from ranking depth
        has_ranking = bool(ranking.get("metric_ranks"))
        has_composite = bool(composite.get("overall_score"))
        has_position = bool(payload.get("market_position"))
        completeness = _clamp01(
            (0.4 if has_ranking else 0.0)
            + (0.4 if has_composite else 0.0)
            + (0.2 if has_position else 0.0)
        )

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
                "source": "benchmark_node",
                "has_ranking": has_ranking,
                "has_composite": has_composite,
                "has_market_position": has_position,
                "market_position": (
                    payload.get("market_position", {}).get("position")
                    if isinstance(payload.get("market_position"), Mapping)
                    else None
                ),
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
                gate_status=GateStatus.DEGRADE,
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
        cohort_gate = GateStatus.PASS if score > 0.2 else GateStatus.DEGRADE

        return LayerIntegrity(
            available=True,
            coverage_ratio=coverage_ratio,
            source_reliability=source_reliability,
            completeness=completeness,
            score=score,
            meta={"status": "success" if score > 0 else "insufficient_data", "cohort_count": cohort_count},
            gate_status=cohort_gate,
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
                gate_status=GateStatus.DEGRADE,
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
        seg_gate = GateStatus.PASS if score > 0.2 else GateStatus.DEGRADE

        return LayerIntegrity(
            available=True,
            coverage_ratio=coverage_ratio,
            source_reliability=source_reliability,
            completeness=completeness,
            score=score,
            meta={"status": "success" if score > 0 else "insufficient_data", "segment_count": segment_count},
            gate_status=seg_gate,
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
