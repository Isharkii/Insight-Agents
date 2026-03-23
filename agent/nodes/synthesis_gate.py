"""
agent/nodes/synthesis_gate.py

Pre-LLM validation gate that hard-blocks insight generation when
required deterministic signals are unavailable or confidence is
below the minimum threshold.

This enforces the architectural contract:
    No LLM reasoning without validated deterministic signals.
"""

from __future__ import annotations

from typing import Any

from agent.graph_config import graph_node_config_for_business_type, signal_name_for_state_key
from agent.nodes.node_result import payload_of, status_of
from agent.signal_integrity import UnifiedSignalIntegrity
from agent.state import AgentState
from app.services.statistics.signal_conflict import apply_conflict_penalty
from llm_synthesis.schema import InsightOutput, set_self_analysis_mode

# Minimum deterministic confidence to allow LLM synthesis.
# Below this threshold the pipeline returns a structured failure.
MIN_CONFIDENCE_FOR_SYNTHESIS = 0.4

# ── Hard-gating thresholds ─────────────────────────────────────────────
_MIN_DEPTH = 3          # minimum KPI time-series data points
_MIN_VALID_LAYERS = 3   # minimum analytical layers with usable signal
_MIN_KPI_COVERAGE = 0.5 # at least 50% of expected metrics present


def _collect_required_failures(state: AgentState) -> list[str]:
    """Return names of required signals that hard-failed.

    Signals with ``insufficient_data`` or ``partial`` status are NOT failures —
    they produced best-effort output and should be gated by confidence (Gate 3),
    not by status (Gate 2).  Only truly absent or crashed signals block here.
    """
    config = graph_node_config_for_business_type(
        str(state.get("business_type") or ""),
    )
    # Statuses that represent a hard failure (no usable output produced).
    _HARD_FAIL_STATUSES = {"failed", "skipped"}

    failures: list[str] = []
    for key in config.required:
        value = state.get(key)
        if value is None:
            failures.append(signal_name_for_state_key(key))
        elif status_of(value) in _HARD_FAIL_STATUSES:
            failures.append(signal_name_for_state_key(key))
    return failures


def _compute_pre_synthesis_confidence(state: AgentState) -> float:
    """Compute deterministic confidence from signal envelopes before LLM runs."""
    integrity = UnifiedSignalIntegrity.compute(state)
    confidence = max(0.0, min(1.0, float(integrity.get("overall_score") or 0.0)))

    conflict_payload = payload_of(state.get("signal_conflicts")) or {}
    conflict_result = conflict_payload.get("conflict_result")
    if isinstance(conflict_result, dict):
        adjusted = apply_conflict_penalty(confidence, conflict_result, floor=0.0)
        confidence = max(
            0.0,
            min(1.0, float(adjusted.get("adjusted_confidence") or confidence)),
        )
    return confidence


# Minimum conflict severity threshold that blocks synthesis outright.
# Any conflict with uncertainty_flag=True (severity >= 0.8) AND
# total_severity above this threshold means the signal landscape is
# too contradictory for reliable narrative generation.
_HIGH_CONFLICT_SEVERITY_THRESHOLD = 1.0


def _has_blocking_conflicts(state: AgentState) -> bool:
    """Return True if signal conflicts are severe enough to block synthesis."""
    conflict_payload = payload_of(state.get("signal_conflicts")) or {}
    conflict_result = conflict_payload.get("conflict_result")
    if not isinstance(conflict_result, dict):
        return False
    uncertainty_flag = bool(conflict_result.get("uncertainty_flag", False))
    total_severity = float(conflict_result.get("total_severity", 0.0))
    conflict_count = int(conflict_result.get("conflict_count", 0))
    # Block if: high-severity conflict exists AND cumulative severity
    # exceeds threshold, OR 3+ conflicts regardless of individual severity.
    if uncertainty_flag and total_severity >= _HIGH_CONFLICT_SEVERITY_THRESHOLD:
        return True
    if conflict_count >= 3:
        return True
    return False


def pre_synthesis_audit(state: AgentState) -> dict[str, Any]:
    """Hard-gating audit that checks minimum data requirements.

    Unit-testable function.  Returns structured result::

        {
            "status": "pass" | "blocked" | "degraded",
            "reason": "...",
            "missing_requirements": [...],
            "uncertainty_mode": bool,
            "conflict_summary": {...} | None,
            "audit_details": {...}
        }
    """
    from app.services.statistics.signal_conflict import build_uncertainty_summary

    integrity = UnifiedSignalIntegrity.compute(state)
    hard_blocks: list[str] = []
    soft_blocks: list[str] = []
    reasons: list[str] = []

    kpi_depth = int(integrity.get("kpi_depth", 0))
    valid_layer_count = int(integrity.get("valid_layer_count", 0))
    kpi_coverage = float(integrity.get("kpi_coverage_ratio", 0.0))
    forecast_usable = bool(integrity.get("forecast_usable", False))

    # Gate 5: depth check
    # depth == 0 → hard block (no time-series data at all)
    # depth 1-2  → soft block (some data, allow degraded synthesis)
    # depth >= 3 → pass
    if kpi_depth == 0:
        hard_blocks.append(
            f"no time-series data points available (depth={kpi_depth})"
        )
        reasons.append("depth=0")
    elif kpi_depth < _MIN_DEPTH:
        soft_blocks.append(
            f"limited data depth ({kpi_depth} < {_MIN_DEPTH} ideal)"
        )
        reasons.append(f"depth={kpi_depth}")

    # Gate 6: layer count check
    # 0 valid layers → hard block (no usable signal at all)
    # 1-2 valid layers → partial_insight (synthesis allowed but degraded)
    # 3+ valid layers → full_insight
    if valid_layer_count == 0:
        hard_blocks.append(
            "no valid analytical layers available"
        )
        reasons.append("valid_layers=0")

    # Gate 7: KPI coverage check
    # 0% coverage → hard block (no metrics at all)
    # > 0% but < 50% → soft block (some metrics, allow degraded)
    # >= 50% → pass
    if kpi_coverage == 0.0:
        hard_blocks.append("no KPI metrics available (coverage=0%)")
        reasons.append("kpi_coverage=0.00")
    elif kpi_coverage < _MIN_KPI_COVERAGE:
        soft_blocks.append(
            f"incomplete KPI coverage "
            f"({kpi_coverage:.0%} < {_MIN_KPI_COVERAGE:.0%} ideal)"
        )
        reasons.append(f"kpi_coverage={kpi_coverage:.2f}")

    # Gate 8: forecast unusable AND zero valid layers
    # 1-2 valid layers → partial_insight (handled by insight_quality), not blocked
    if not forecast_usable and valid_layer_count == 0:
        hard_blocks.append(
            "forecast unusable with no alternative signal coverage"
        )
        reasons.append("forecast_unusable+no_alternatives")

    # Uncertainty mode from conflict severity
    conflict_payload = payload_of(state.get("signal_conflicts")) or {}
    conflict_result = conflict_payload.get("conflict_result")
    uncertainty_mode = False
    conflict_summary = None
    if isinstance(conflict_result, dict):
        total_severity = float(conflict_result.get("total_severity", 0.0))
        if total_severity > 1.0:
            uncertainty_mode = True
            conflict_summary = build_uncertainty_summary(conflict_result)

    # Insight quality from integrity model
    insight_quality = str(integrity.get("insight_quality", "blocked"))

    # Combine for backward-compatible missing_requirements list
    missing_requirements = hard_blocks + soft_blocks

    if hard_blocks:
        status = "blocked"
        reason = "; ".join(reasons)
        insight_quality = "blocked"
    elif soft_blocks:
        # Soft blocks allow degraded synthesis with hedged tone —
        # the data is sparse but not absent.
        status = "degraded"
        reason = "; ".join(reasons)
        if valid_layer_count > 0:
            insight_quality = "partial_insight"
    elif uncertainty_mode:
        status = "degraded"
        reason = "conflicting signals detected (uncertainty_mode active)"
    elif insight_quality == "partial_insight":
        status = "degraded"
        reason = (
            f"partial insight — only {valid_layer_count} valid layer(s) "
            f"(minimum {_MIN_VALID_LAYERS} for full insight)"
        )
    else:
        status = "pass"
        reason = "all requirements met"

    return {
        "status": status,
        "reason": reason,
        "missing_requirements": missing_requirements,
        "uncertainty_mode": uncertainty_mode,
        "insight_quality": insight_quality,
        "conflict_summary": conflict_summary,
        "audit_details": {
            "kpi_depth": kpi_depth,
            "valid_layer_count": valid_layer_count,
            "kpi_coverage_ratio": round(kpi_coverage, 4),
            "forecast_usable": forecast_usable,
            "overall_confidence": integrity.get("overall_score", 0.0),
            "layer_classification": integrity.get("layer_classification", {}),
        },
    }


def should_block_synthesis(state: AgentState) -> bool:
    """Return True if the pipeline must NOT proceed to LLM synthesis."""
    import logging as _logging
    _block_log = _logging.getLogger("agent.nodes.synthesis_gate.block_check")

    # Gate 1: pipeline_status explicitly failed
    pipeline_status = str(state.get("pipeline_status") or "").strip().lower()
    if pipeline_status == "failed":
        _block_log.info("Gate 1 BLOCKED: pipeline_status == failed")
        return True

    # Gate 2: any required signal missing or failed
    required_failures = _collect_required_failures(state)
    if required_failures:
        _block_log.info("Gate 2 BLOCKED: required signals failed: %s", required_failures)
        return True

    # Gates 5-8: hard-gating audit (depth, layers, coverage, forecast)
    # Run audit early so insight_quality is available for Gate 3 and 4.
    audit = pre_synthesis_audit(state)
    if audit["status"] == "blocked":
        _block_log.info("Gates 5-8 BLOCKED: audit=%s", audit.get("reason"))
        return True

    is_partial = audit.get("insight_quality") == "partial_insight"
    _block_log.info(
        "Gate check: insight_quality=%s, is_partial=%s, audit_status=%s",
        audit.get("insight_quality"), is_partial, audit["status"],
    )

    # Gate 3: deterministic confidence below threshold
    # partial_insight states have confidence capped at 0.25 by design —
    # they are allowed through (degraded, not blocked).
    if not is_partial:
        pre_conf = _compute_pre_synthesis_confidence(state)
        if pre_conf < MIN_CONFIDENCE_FOR_SYNTHESIS:
            _block_log.info("Gate 3 BLOCKED: confidence %.3f < %.2f", pre_conf, MIN_CONFIDENCE_FOR_SYNTHESIS)
            return True

    # Gate 4: signal conflicts too severe for reliable narrative
    # partial_insight states already carry hedged tone; blocking them
    # on conflict count alone prevents any synthesis from sparse data.
    if not is_partial and _has_blocking_conflicts(state):
        _block_log.info("Gate 4 BLOCKED: severe signal conflicts")
        return True

    _block_log.info("All gates PASSED — synthesis allowed")
    return False


def _has_competitor_data(state: AgentState) -> bool:
    """Read the deterministic competitive context contract from state."""
    ctx = state.get("competitive_context")
    if not isinstance(ctx, dict):
        return False
    return bool(ctx.get("available", False))


def build_blocked_response(state: AgentState) -> str:
    """Build a structured failure response when synthesis is blocked.

    Returns the serialized JSON string for ``state["final_response"]``.
    Sets self-analysis mode so the fallback text matches the data context.
    """
    has_competitors = _has_competitor_data(state)
    set_self_analysis_mode(not has_competitors)

    try:
        failed_signals = _collect_required_failures(state)
        pipeline_status = str(state.get("pipeline_status") or "failed").strip().lower()
        pre_confidence = _compute_pre_synthesis_confidence(state)

        # Run the audit for structured missing_requirements
        try:
            audit = pre_synthesis_audit(state)
        except Exception:
            audit = {"missing_requirements": [], "uncertainty_mode": False}

        parts: list[str] = []
        if failed_signals:
            parts.append(
                f"required signal(s) unavailable: {', '.join(failed_signals)}"
            )
        if pre_confidence < MIN_CONFIDENCE_FOR_SYNTHESIS:
            parts.append(
                f"deterministic confidence too low ({pre_confidence:.2f} < "
                f"{MIN_CONFIDENCE_FOR_SYNTHESIS})"
            )
        for req in audit.get("missing_requirements", []):
            parts.append(req)
        reason = (
            f"Insight generation blocked: {'; '.join(parts)}. "
            f"Pipeline status: {pipeline_status}. "
            f"The system requires validated deterministic signals "
            f"above the confidence threshold before LLM synthesis can execute."
        )

        failure = InsightOutput.failure(
            reason=reason,
            pipeline_status=pipeline_status if pipeline_status in ("success", "partial", "failed") else "failed",
        )

        return failure.model_dump_json()
    finally:
        set_self_analysis_mode(False)


def synthesis_gate_node(state: AgentState) -> AgentState:
    """LangGraph node: blocks LLM synthesis when required signals failed.

    When blocked, writes a structured failure to ``final_response`` and
    sets ``synthesis_blocked = True`` so the conditional edge can route
    directly to END, skipping the LLM node entirely.

    This node MUST NOT raise — if it does, the wrapper returns ``{}``
    and ``final_response`` stays None, causing the frontend to show a
    generic "Pipeline did not produce final_response" error.
    """
    import logging as _logging

    _gate_logger = _logging.getLogger("agent.nodes.synthesis_gate")

    # Compute integrity safely — fall back to empty dict on failure.
    try:
        integrity = UnifiedSignalIntegrity.compute(state)
    except Exception as exc:
        _gate_logger.warning("synthesis_gate: integrity computation failed: %s", exc)
        integrity = {}

    try:
        blocked = should_block_synthesis(state)
    except Exception as exc:
        _gate_logger.warning("synthesis_gate: block check failed, defaulting to blocked: %s", exc)
        blocked = True

    if blocked:
        # Build the blocked response safely — a failure here previously
        # caused the wrapper to swallow the exception and return {},
        # losing final_response entirely.
        try:
            final_response = build_blocked_response(state)
        except Exception as exc:
            _gate_logger.error("synthesis_gate: build_blocked_response failed: %s", exc)
            # Last-resort fallback: build a minimal valid InsightOutput
            final_response = InsightOutput.failure(
                f"Synthesis gate internal error: {exc}"
            ).model_dump_json()

        try:
            integrity_scores = UnifiedSignalIntegrity.score_vector_from_integrity(integrity)
        except Exception:
            integrity_scores = {}

        try:
            pre_confidence = _compute_pre_synthesis_confidence(state)
        except Exception:
            pre_confidence = 0.0

        conflict_payload = payload_of(state.get("signal_conflicts")) or {}
        conflict_result = conflict_payload.get("conflict_result")
        warnings = ["Synthesis blocked by deterministic pre-checks."]
        warnings.extend(integrity.get("reasoning_warnings", []))

        # Compute audit for structured diagnostics
        try:
            audit = pre_synthesis_audit(state)
        except Exception:
            audit = {
                "status": "blocked",
                "missing_requirements": [],
                "uncertainty_mode": False,
                "conflict_summary": None,
            }

        # ── Build spec-compliant block_reasons list ─────────────────
        block_reasons: list[str] = []
        _ps = str(state.get("pipeline_status") or "").strip().lower()
        if _ps == "failed":
            block_reasons.append("pipeline_status == failed")
        for sig in _collect_required_failures(state):
            block_reasons.append(f"required signal missing: {sig}")
        for req in audit.get("missing_requirements", []):
            block_reasons.append(req)
        if pre_confidence < MIN_CONFIDENCE_FOR_SYNTHESIS and audit.get("insight_quality") != "partial_insight":
            block_reasons.append(
                f"confidence below threshold ({pre_confidence:.2f} < {MIN_CONFIDENCE_FOR_SYNTHESIS})"
            )
        if _has_blocking_conflicts(state) and audit.get("insight_quality") != "partial_insight":
            block_reasons.append("signal conflicts too severe (uncertainty_flag + severity >= 1.0)")
        if not block_reasons:
            block_reasons.append("synthesis blocked by deterministic pre-checks")

        return {
            "synthesis_blocked": True,
            "pipeline_status": "blocked",
            "eligible_for_llm": False,
            "block_reasons": block_reasons,
            "final_response": final_response,
            "signal_integrity": integrity,
            "envelope_diagnostics": {
                "warnings": warnings,
                "missing_signal": _collect_required_failures(state),
                "missing_requirements": audit.get("missing_requirements", []),
                "insight_quality": audit.get("insight_quality", "blocked"),
                "uncertainty_mode": audit.get("uncertainty_mode", False),
                "conflict_summary": audit.get("conflict_summary"),
                "layer_classification": integrity.get("layer_classification", {}),
                "isolated_layers": integrity.get("isolated_layers", []),
                "degraded_layers": integrity.get("degraded_layers", []),
                "reasoning_warnings": integrity.get("reasoning_warnings", []),
                "confidence_adjustments": integrity.get("confidence_adjustments", []),
                "confidence_breakdown": integrity.get("confidence_breakdown", {}),
                "confidence_score": pre_confidence,
                "signal_conflicts": (
                    conflict_result if isinstance(conflict_result, dict) else {}
                ),
                "signal_integrity_scores": integrity_scores,
                "signal_integrity": integrity,
                "missing_signal_report": integrity.get("missing_signal_report", {}),
                "kpi_coverage_report": integrity.get("kpi_coverage_report", {}),
                "benchmark_status": {
                    "available": status_of(state.get("benchmark_data")) == "success",
                    "status": status_of(state.get("benchmark_data")) or "missing",
                    "competitive_context_available": bool(
                        (state.get("competitive_context") or {}).get("available", False)
                    ),
                },
            },
        }

    return {
        "synthesis_blocked": False,
        "eligible_for_llm": True,
        "signal_integrity": integrity,
    }
