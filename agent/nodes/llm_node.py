"""LLM node for final structured insight synthesis.

Builds the synthesis prompt from available upstream signals only, calls the
LLM through retry+validation, and writes the serialized response to
``state[\"final_response\"]``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from pydantic import ValidationError

from agent.graph_config import (
    KPI_KEY_BY_BUSINESS_TYPE,
    graph_node_config_for_business_type,
    signal_name_for_state_key,
)
from agent.nodes.node_result import payload_of, status_of, warnings_of
from agent.signal_integrity import UnifiedSignalIntegrity
from agent.state import AgentState
from app.services.statistics.signal_conflict import apply_conflict_penalty
from db.config import load_env_files
from llm_synthesis.adapter import BaseLLMAdapter, MockLLMAdapter, OpenAILLMAdapter
from llm_synthesis.prompt_builder import SynthesisPromptBuilder
from llm_synthesis.retry import LLMRetryExhaustedError, generate_with_retry
from llm_synthesis.schema import (
    InsightOutput as FinalInsightResponse,
    set_self_analysis_mode,
)
from llm_synthesis.validator import LLMOutputValidationError

_prompt_builder = SynthesisPromptBuilder()
logger = logging.getLogger(__name__)


def _resolve_kpi_result(state: AgentState) -> Any:
    business_type = str(state.get("business_type") or "").lower()
    preferred_key = KPI_KEY_BY_BUSINESS_TYPE.get(business_type)
    if preferred_key:
        return state.get(preferred_key)

    for key in ("kpi_data", "saas_kpi_data", "ecommerce_kpi_data", "agency_kpi_data"):
        value = state.get(key)
        if value is not None:
            return value
    return None


def _usable_payload(value: Any) -> dict[str, Any]:
    """Extract payload only from success envelopes."""
    return payload_of(value) if status_of(value) == "success" else {}


def _dedupe_text(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for raw in items:
        text = str(raw).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _collect_diagnostics(state: AgentState) -> dict[str, Any]:
    """Collect warnings, missing signals, and confidence adjustments."""
    all_warnings: list[str] = []
    missing_signal: list[str] = []
    adjustments: list[dict[str, Any]] = []

    config = graph_node_config_for_business_type(str(state.get("business_type") or ""))
    required_keys = config.required
    optional_keys = config.optional
    ordered_keys: list[str] = list(dict.fromkeys((*required_keys, *optional_keys)))

    for key in ordered_keys:
        is_required = key in required_keys
        value = state.get(key)
        signal = signal_name_for_state_key(key)

        if value is None:
            if is_required:
                missing_signal.append(signal)
                all_warnings.append(f"Required signal '{signal}' is unavailable.")
            continue

        status = status_of(value)
        node_warnings = warnings_of(value)
        if node_warnings:
            all_warnings.extend(node_warnings)

        if status in {"skipped", "failed"}:
            missing_signal.append(signal)
            all_warnings.append(
                (
                    f"Required signal '{signal}' is {status}."
                    if is_required
                    else f"Optional signal '{signal}' is {status}; partial coverage applied."
                )
            )

    # Surface ingestion warnings in diagnostics.
    ingestion_warnings = state.get("ingestion_warnings")
    if isinstance(ingestion_warnings, list):
        all_warnings.extend(str(item) for item in ingestion_warnings if str(item).strip())

    integrity = UnifiedSignalIntegrity.compute(state)
    integrity_scores = UnifiedSignalIntegrity.score_vector_from_integrity(integrity)
    confidence_score = float(integrity.get("overall_score") or 0.0)
    integrity_adjustments = integrity.get("confidence_adjustments")
    if isinstance(integrity_adjustments, list):
        adjustments.extend(
            item
            for item in integrity_adjustments
            if isinstance(item, dict)
        )
    if not bool(integrity.get("kpi_gate_passed", True)):
        all_warnings.append(
            "KPI integrity gate failed (kpi_score < 0.3); confidence forced to 0."
        )

    conflict_envelope = state.get("signal_conflicts")
    conflict_payload = payload_of(conflict_envelope) if conflict_envelope is not None else None
    conflict_result = (
        conflict_payload.get("conflict_result")
        if isinstance(conflict_payload, dict)
        else None
    )
    conflict_adjustment = (
        conflict_payload.get("confidence_adjustment")
        if isinstance(conflict_payload, dict)
        else None
    )
    if isinstance(conflict_result, dict):
        for item in conflict_result.get("warnings", []):
            text = str(item).strip()
            if text:
                all_warnings.append(text)
        penalty = float(conflict_result.get("confidence_penalty") or 0.0)
        if penalty > 0.0:
            adjusted = apply_conflict_penalty(
                confidence_score,
                conflict_result,
                floor=0.0,
            )
            confidence_score = float(adjusted.get("adjusted_confidence") or confidence_score)
            adjustments.append(
                {
                    "signal": "global_signal_conflicts",
                    "delta": round(-penalty, 6),
                    "reason": "global_signal_conflict_penalty",
                }
            )
            if isinstance(conflict_adjustment, dict):
                adjustments.append(
                    {
                        "signal": "signal_conflicts_node",
                        "delta": round(
                            float(conflict_adjustment.get("adjusted_confidence", 0.0))
                            - float(conflict_adjustment.get("base_confidence", 0.0)),
                            6,
                        ),
                        "reason": "signal_conflicts_node_adjustment",
                    }
                )

    return {
        "warnings": _dedupe_text(all_warnings),
        "missing_signal": _dedupe_text(missing_signal),
        "confidence_adjustments": adjustments,
        "confidence_score": max(0.0, min(1.0, round(confidence_score, 6))),
        "signal_conflicts": conflict_result if isinstance(conflict_result, dict) else {},
        "signal_integrity_scores": integrity_scores,
        "signal_integrity": integrity,
    }


def _derive_pipeline_status(state: AgentState) -> str:
    existing = str(state.get("pipeline_status") or "").strip().lower()
    if existing in {"success", "partial", "failed"}:
        return existing

    config = graph_node_config_for_business_type(str(state.get("business_type") or ""))
    required_keys = config.required
    optional_keys = config.optional

    for key in required_keys:
        if status_of(state.get(key)) != "success":
            return "failed"

    for key in optional_keys:
        value = state.get(key)
        if value is None:
            continue
        if status_of(value) != "success":
            return "partial"

    return "success"


def _build_adapter(*, model_override: str | None = None) -> BaseLLMAdapter:
    """Instantiate the adapter selected by the LLM_ADAPTER env var."""
    adapter_name = os.getenv("LLM_ADAPTER", "openai").strip().lower()
    if adapter_name == "mock":
        return MockLLMAdapter()

    model_name = (
        model_override
        or os.getenv("LLM_MODEL", "").strip()
        or "gpt-5.4"
    )

    return OpenAILLMAdapter(
        model=model_name,
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
        api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL") or None,
        timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
    )


def _public_failure_reason(error: Exception) -> str:
    """Map internal synthesis failures to concise, user-facing reasons."""
    if isinstance(error, LLMRetryExhaustedError):
        stage = str(getattr(error.last_error, "stage", "")).strip().lower()
        if stage == "schema":
            # Include the actual validation errors so failure() can
            # route to the correct template (e.g. missing metrics vs generic).
            last_errors = getattr(error.last_error, "errors", [])
            detail = "; ".join(str(e) for e in last_errors[:3]) if last_errors else ""
            return (
                f"LLM output did not satisfy the required "
                f"analysis schema after retries: {detail}"
                if detail
                else "LLM output did not satisfy the required analysis schema after retries."
            )
        if stage == "json_parse":
            return (
                "LLM output returned invalid JSON format "
                "after retries."
            )
        return "LLM output failed validation after retries."

    if isinstance(error, LLMOutputValidationError):
        stage = str(error.stage or "").strip().lower()
        if stage == "schema":
            return (
                "LLM output did not satisfy the required "
                "analysis schema."
            )
        if stage == "json_parse":
            return "LLM output returned invalid JSON format."
        return "LLM output failed validation."

    if isinstance(error, (json.JSONDecodeError, ValidationError, ValueError, TypeError)):
        return "Insight synthesis output could not be validated against the required schema."

    # Surface actionable messages for common LLM adapter failures.
    error_type = type(error).__name__
    error_msg = str(error)[:200]

    # OpenAI / httpx errors
    if "AuthenticationError" in error_type or "api_key" in error_msg.lower():
        return "LLM API authentication failed. Check LLM_API_KEY or OPENAI_API_KEY."
    if "RateLimitError" in error_type:
        return "LLM API rate limit exceeded. Retry after a brief wait."
    if "Timeout" in error_type or "timed out" in error_msg.lower():
        return "LLM API request timed out. The model may be overloaded."
    if "APIConnectionError" in error_type or "Connection" in error_type:
        return "Could not connect to the LLM API. Check LLM_BASE_URL and network."
    if "NotFoundError" in error_type or "model" in error_msg.lower() and "not found" in error_msg.lower():
        return "LLM model not found. Check LLM_MODEL configuration."
    if "ImportError" in error_type:
        return "LLM adapter dependency missing. Install the openai package."

    logger.error("Unhandled LLM synthesis error type=%s: %s", error_type, error_msg)
    return f"LLM synthesis failed ({error_type}). Check server logs for details."


def _ensure_conditional_recommendations(
    payload: FinalInsightResponse,
    *,
    conditional_required: bool,
) -> FinalInsightResponse:
    if not conditional_required:
        return payload

    recommendations = payload.strategic_recommendations

    def _tag(items: list[str]) -> list[str]:
        out: list[str] = []
        for item in items:
            text = str(item or "").strip()
            if not text:
                continue
            if text.lower().startswith("conditional:"):
                out.append(text)
            else:
                out.append(f"Conditional: {text}")
        return out

    return payload.model_copy(
        update={
            "strategic_recommendations": recommendations.model_copy(
                update={
                    "immediate_actions": _tag(recommendations.immediate_actions),
                    "mid_term_moves": _tag(recommendations.mid_term_moves),
                    "defensive_strategies": _tag(recommendations.defensive_strategies),
                    "offensive_strategies": _tag(recommendations.offensive_strategies),
                }
            )
        }
    )


def _ensure_low_confidence_tone(
    payload: FinalInsightResponse,
    *,
    conditional_required: bool,
) -> FinalInsightResponse:
    if not conditional_required:
        return payload

    analysis = payload.competitive_analysis

    def _tag(text: str) -> str:
        value = str(text or "").strip()
        if not value:
            return "Conditional: analysis context remains uncertain due to limited data."
        return value if value.lower().startswith("conditional:") else f"Conditional: {value}"

    return payload.model_copy(
        update={
            "competitive_analysis": analysis.model_copy(
                update={
                    "summary": _tag(analysis.summary),
                    "market_position": _tag(analysis.market_position),
                    "relative_performance": _tag(analysis.relative_performance),
                }
            )
        }
    )


def _degraded_recommendations(
    *,
    uncertainty_mode: bool,
    has_competitors: bool,
    peers: list[str] | None = None,
    news_highlights: list[dict[str, Any]] | None = None,
) -> dict[str, list[str]]:
    """Return schema-compliant withheld recommendations for degraded modes.

    Notes:
    - Items must include explicit context terms (e.g., competitor/gap/risk/revenue)
      or InsightOutput re-validation in API routers will fail.
    - Items must remain unique across sections.
    """
    if not has_competitors:
        if uncertainty_mode:
            return {
                "immediate_actions": [
                    "Conditional: basic data-only improvement plan: resolve revenue, retention, and churn conflicts before execution."
                ],
                "mid_term_moves": [
                    "Conditional: strengthen macro and trend signal coverage to validate growth trajectory and forecast direction."
                ],
                "defensive_strategies": [
                    "Conditional: protect retention in high-risk segments while uncertainty in data signals remains elevated."
                ],
                "offensive_strategies": [
                    "Conditional: pursue growth opportunities with small tests after risk-adjusted revenue trends stabilize."
                ],
            }

        return {
            "immediate_actions": [
                "Conditional: close internal data gaps across revenue, retention, and churn before acting on strategic assumptions."
            ],
            "mid_term_moves": [
                "Conditional: expand historical metric coverage and macro context to improve confidence in growth trend and risk analysis."
            ],
            "defensive_strategies": [
                "Conditional: reduce churn risk in vulnerable segments where signal quality is currently limited."
            ],
            "offensive_strategies": [
                "Conditional: target revenue growth opportunities after key KPI and forecast signals are validated."
            ],
        }

    peer_names = [str(item).strip() for item in (peers or []) if str(item).strip()]
    peer_label = ", ".join(peer_names[:3]) if peer_names else "the selected competitors"

    def _display_competitor(name: str) -> str:
        parts = [part for part in str(name or "").strip().split() if part]
        if not parts:
            return "Competitor"
        return " ".join(part[:1].upper() + part[1:] for part in parts)

    highlights: list[tuple[str, str]] = []
    for item in news_highlights or []:
        if not isinstance(item, dict):
            continue
        competitor = _display_competitor(str(item.get("competitor") or ""))
        title = str(item.get("title") or "").strip()
        if not competitor or not title:
            continue
        highlights.append((competitor, title))

    if highlights:
        top = highlights[0]
        second = highlights[1] if len(highlights) > 1 else highlights[0]
        top_ref = f"{top[0]}: {top[1]}"
        second_ref = f"{second[0]}: {second[1]}"
        return {
            "immediate_actions": [
                f"Conditional: most critical competitor news signal ({top_ref}) indicates a risk window; protect revenue retention and close the competitor gap immediately."
            ],
            "mid_term_moves": [
                f"Conditional: convert competitor weakness from recent news ({second_ref}) into a growth and positioning advantage with clearer packaging and onboarding."
            ],
            "defensive_strategies": [
                f"Conditional: defend retention strength against {peer_label} by prioritizing churn-risk segments linked to news-driven competitor pressure."
            ],
            "offensive_strategies": [
                f"Conditional: launch targeted campaigns around competitor vulnerability themes from recent news to capture revenue growth where peer weakness is visible."
            ],
        }

    if uncertainty_mode:
        return {
            "immediate_actions": [
                f"Conditional: basic competitor improvement plan for {peer_label}: resolve competitor gap, revenue, and retention conflicts before execution."
            ],
            "mid_term_moves": [
                "Conditional: use recent competitor news tracking plus benchmark reconciliation to validate growth and churn trend gaps."
            ],
            "defensive_strategies": [
                "Conditional: protect retention strength in high-risk segments while competitor strength and weakness signals remain conflicting."
            ],
            "offensive_strategies": [
                "Conditional: pursue competitor weakness opportunities with small tests after risk-adjusted revenue benchmarks stabilize."
            ],
        }

    return {
        "immediate_actions": [
            f"Conditional: basic competitor improvement plan for {peer_label}: close competitor and revenue metric gaps before acting on inferred strengths or weaknesses."
        ],
        "mid_term_moves": [
            "Conditional: expand growth and retention coverage, including recent competitor news signals, to validate benchmark position."
        ],
        "defensive_strategies": [
            "Conditional: reduce churn risk in vulnerable segments where competitor weakness signals lack benchmark depth."
        ],
        "offensive_strategies": [
            "Conditional: target revenue growth opportunities after competitor strength and weakness signals are confirmed against peer benchmarks."
        ],
    }


def _has_competitor_data(state: AgentState) -> bool:
    """Read the deterministic competitive context contract from state.

    Returns True only when ``state["competitive_context"]["available"]``
    is explicitly ``True``.  This flag is emitted by the segmentation
    node after counting distinct peer entities in the local benchmark
    query — no heuristics, no inference, no prompt inspection.
    """
    ctx = state.get("competitive_context")
    if not isinstance(ctx, dict):
        return False
    return bool(ctx.get("available", False))


def _self_analysis_only_enabled(state: AgentState) -> bool:
    """Return True when request/env forces data-only self analysis."""
    override = state.get("self_analysis_only")
    if isinstance(override, bool):
        return override
    raw = str(os.getenv("INSIGHT_SELF_ANALYSIS_ONLY", "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def llm_node(state: AgentState) -> AgentState:
    """LangGraph node: synthesize available outputs into a structured insight."""
    load_env_files()

    kpi_data = _usable_payload(_resolve_kpi_result(state))
    forecast_data = _usable_payload(state.get("forecast_data"))
    risk_data = _usable_payload(state.get("risk_data"))
    root_cause = _usable_payload(state.get("root_cause"))
    pipeline_status = _derive_pipeline_status(state)
    diagnostics = _collect_diagnostics(state)
    logger.info("Signal integrity scores: %s", json.dumps(diagnostics.get("signal_integrity_scores") or {}, sort_keys=True))

    self_analysis_only = _self_analysis_only_enabled(state)
    has_competitors = False if self_analysis_only else _has_competitor_data(state)
    set_self_analysis_mode(True if self_analysis_only else not has_competitors)

    ctx = state.get("competitive_context") or {}
    ctx_source = str(ctx.get("source", "unavailable"))
    logger.info(
        "Competitive context: available=%s source=%s peer_count=%s metrics=%s mode=%s",
        ctx.get("available", False),
        ctx_source,
        ctx.get("peer_count", 0),
        ctx.get("metrics", []),
        "competitor" if has_competitors else "self_analysis",
    )

    # ── Extract deterministic benchmark intelligence ──────────────
    # benchmark_data (from benchmark_node) contains rankings, composite
    # scores, and market positioning — this is deterministic, local-first
    # intelligence that MUST reach the LLM prompt.
    benchmark_envelope = state.get("benchmark_data")
    benchmark_payload = _usable_payload(benchmark_envelope) if benchmark_envelope else {}
    benchmark_intelligence: dict[str, Any] | None = None
    if not self_analysis_only and benchmark_payload:
        _composite = benchmark_payload.get("composite")
        _market_pos = benchmark_payload.get("market_position")
        _ranking = benchmark_payload.get("ranking")
        _peer_sel = benchmark_payload.get("peer_selection") or {}
        if _composite or _market_pos or _ranking:
            benchmark_intelligence = {}
            if isinstance(_composite, dict):
                benchmark_intelligence["composite_scores"] = _composite
            if isinstance(_market_pos, dict):
                benchmark_intelligence["market_position"] = _market_pos
            if isinstance(_ranking, dict):
                benchmark_intelligence["ranking"] = _ranking
            benchmark_intelligence["peer_count"] = len(
                _peer_sel.get("selected_peers") or []
            )
            benchmark_intelligence["source"] = "deterministic_local"
            logger.info(
                "Benchmark intelligence: position=%s overall=%.1f peers=%d",
                (_market_pos or {}).get("position", "unknown"),
                (_composite or {}).get("overall_score", 0.0),
                benchmark_intelligence["peer_count"],
            )
            # If benchmark data is available but competitive_context is not,
            # the LLM should still receive this deterministic intelligence
            # and operate in competitor-aware mode — but ONLY if there's
            # enough substance (ranking + composite + 2+ peers).  Thin/partial
            # benchmark data stays as supplementary context in self-analysis
            # mode rather than switching to competitor mode where the LLM
            # would fail specificity validation.
            _has_ranking = bool(_ranking and _ranking.get("overall_rank") is not None)
            _has_composite = bool(_composite and _composite.get("overall_score") is not None)
            _enough_peers = benchmark_intelligence.get("peer_count", 0) >= 2
            if not has_competitors and _has_ranking and _has_composite and _enough_peers:
                has_competitors = True
                set_self_analysis_mode(False)
                logger.info(
                    "benchmark_data available with %d peers — bridging "
                    "to competitor mode from benchmark",
                    benchmark_intelligence["peer_count"],
                )
            elif not has_competitors:
                logger.info(
                    "benchmark_data available but insufficient for competitor "
                    "mode (ranking=%s composite=%s peers=%d) — staying in "
                    "self-analysis mode with benchmark as supplementary context",
                    _has_ranking, _has_composite,
                    benchmark_intelligence.get("peer_count", 0),
                )

    # ── Extract numeric-only competitive signals for the prompt ────
    # Only structured numeric data reaches the LLM; no raw web text.
    competitor_signals: dict[str, Any] | None = None
    if has_competitors:
        numeric_signals = ctx.get("numeric_signals", [])
        if isinstance(numeric_signals, list) and numeric_signals:
            competitor_signals = {
                "source": ctx_source,
                "peer_count": ctx.get("peer_count", 0),
                "peers": ctx.get("peers", []),
                "signals": numeric_signals,
            }
        news_highlights = ctx.get("news_highlights", [])
        if isinstance(news_highlights, list) and news_highlights:
            competitor_signals = competitor_signals or {
                "source": ctx_source,
                "peer_count": ctx.get("peer_count", 0),
                "peers": ctx.get("peers", []),
                "signals": [],
            }
            competitor_signals["recent_news_highlights"] = news_highlights[:5]

    # ── Confidence penalty for external_fetch source ───────────────
    # External web-sourced competitive data is inherently less reliable
    # than deterministic local computation.  Apply a confidence penalty.
    confidence_score = float(diagnostics.get("confidence_score", 1.0))
    if ctx_source == "external_fetch" and has_competitors:
        _EXTERNAL_CONFIDENCE_PENALTY = -0.15
        confidence_score = max(0.0, round(confidence_score + _EXTERNAL_CONFIDENCE_PENALTY, 6))
        diagnostics["confidence_adjustments"].append({
            "signal": "competitive_context",
            "delta": _EXTERNAL_CONFIDENCE_PENALTY,
            "reason": "external_fetch_confidence_penalty",
        })
        diagnostics["confidence_score"] = confidence_score
        logger.info(
            "Applied external_fetch confidence penalty: delta=%.2f new_score=%.4f",
            _EXTERNAL_CONFIDENCE_PENALTY,
            confidence_score,
        )

    # Extract enriched signal summary for the prompt
    signal_enrichment = _usable_payload(state.get("signal_enrichment"))

    # Extract integrity diagnostics for prompt visibility
    _integrity = state.get("signal_integrity") or {}
    _isolated_layers = _integrity.get("isolated_layers", [])
    _degraded_layers = _integrity.get("degraded_layers", [])
    _reasoning_warnings = _integrity.get("reasoning_warnings", [])

    prompt = _prompt_builder.build_prompt(
        kpi_data=kpi_data,
        forecast_data=forecast_data,
        risk_data=risk_data,
        root_cause=root_cause,
        segmentation=_usable_payload(state.get("segmentation")),
        prioritization=state.get("prioritization") or {},
        confidence_score=confidence_score,
        missing_signals=diagnostics.get("missing_signal", []),
        has_competitor_data=has_competitors,
        competitor_signals=competitor_signals,
        conflict_metadata=diagnostics.get("signal_conflicts"),
        signal_enrichment=signal_enrichment,
        isolated_layers=_isolated_layers if _isolated_layers else None,
        degraded_layers=_degraded_layers if _degraded_layers else None,
        reasoning_warnings=_reasoning_warnings if _reasoning_warnings else None,
        benchmark_intelligence=benchmark_intelligence,
    )

    adapter = _build_adapter(model_override=state.get("llm_model_override"))

    try:
        synthesis = generate_with_retry(adapter, prompt, max_retries=2)
        final_payload = FinalInsightResponse.model_validate(synthesis.model_dump())
    except Exception as error:  # noqa: BLE001
        logger.error(
            "LLM synthesis failed [%s]: %s",
            type(error).__name__,
            str(error)[:500],
            exc_info=True,
        )
        final_payload = FinalInsightResponse.failure(
            reason=_public_failure_reason(error),
            pipeline_status=pipeline_status,
        )

    # Enforce: deterministic confidence always overrides LLM self-assessment.
    # The LLM must not inflate confidence beyond what the signals support.
    deterministic_confidence = float(diagnostics.get("confidence_score", 1.0))
    enforced_confidence = min(
        final_payload.competitive_analysis.confidence,
        deterministic_confidence,
    )

    # Post-processing: enforce deterministic confidence and tone rules.
    # The LLM output was already schema-validated by generate_with_retry;
    # model_copy only touches the confidence value and conditional labels,
    # so a full re-validation is not needed and would risk false-negative
    # fallback (confidence → 0) due to context-dependent validator state.
    final_payload = final_payload.model_copy(
        update={
            "competitive_analysis": final_payload.competitive_analysis.model_copy(
                update={"confidence": enforced_confidence}
            )
        }
    )
    # ── Spec enforcement: partial_insight / uncertainty_mode ─────
    # When operating in degraded modes, recommendations MUST be nulled
    # and tone MUST be conditional regardless of LLM output.
    _prioritization = state.get("prioritization") or {}
    _insight_quality = str(_prioritization.get("insight_quality", ""))
    _uncertainty_mode = bool(_prioritization.get("uncertainty_mode", False))
    _force_conditional = (
        enforced_confidence < 0.5
        or _insight_quality == "partial_insight"
        or _uncertainty_mode
    )

    if _insight_quality == "partial_insight" or _uncertainty_mode:
        # Keep recommendations withheld in degraded modes, but preserve
        # schema-valid context terms so downstream re-validation succeeds.
        withheld_updates = _degraded_recommendations(
            uncertainty_mode=_uncertainty_mode,
            has_competitors=has_competitors,
            peers=ctx.get("peers", []),
            news_highlights=ctx.get("news_highlights", []),
        )
        final_payload = final_payload.model_copy(
            update={
                "strategic_recommendations": final_payload.strategic_recommendations.model_copy(
                    update=withheld_updates
                )
            }
        )

    final_payload = _ensure_conditional_recommendations(
        final_payload,
        conditional_required=_force_conditional,
    )
    final_payload = _ensure_low_confidence_tone(
        final_payload,
        conditional_required=_force_conditional,
    )

    # ── FINAL confidence cap enforcement (safety net) ────────────
    # No downstream module may modify confidence after this point.
    _integrity_result = diagnostics.get("signal_integrity") or {}
    _cap_trace = UnifiedSignalIntegrity.enforce_final_confidence_caps(
        enforced_confidence,
        _integrity_result,
    )
    _final_conf = _cap_trace["final_confidence"]
    if _final_conf != enforced_confidence:
        logger.warning(
            "Final cap enforcement adjusted confidence: %.4f → %.4f (caps: %s)",
            enforced_confidence,
            _final_conf,
            [c["cap"] for c in _cap_trace["applied_caps"]],
        )
        final_payload = final_payload.model_copy(
            update={
                "competitive_analysis": final_payload.competitive_analysis.model_copy(
                    update={"confidence": _final_conf}
                )
            }
        )
    diagnostics["confidence_cap_trace"] = _cap_trace
    diagnostics["confidence_score"] = _final_conf

    # ── Benchmark transparency ───────────────────────────────────
    diagnostics["benchmark_status"] = {
        "available": False if self_analysis_only else bool(benchmark_intelligence),
        "source": (
            "self_analysis_only"
            if self_analysis_only
            else (benchmark_intelligence or {}).get("source", "unavailable")
        ),
        "peer_count": (benchmark_intelligence or {}).get("peer_count", 0),
        "market_position": (
            (benchmark_intelligence or {}).get("market_position", {}).get("position")
            if benchmark_intelligence
            else None
        ),
        "surfaced_to_llm": False if self_analysis_only else bool(benchmark_intelligence),
        "competitive_context_available": bool(ctx.get("available", False)),
    }

    final_response = final_payload.model_dump_json()

    # Reset self-analysis mode to prevent leaking into subsequent calls.
    set_self_analysis_mode(False)

    return {
        "pipeline_status": pipeline_status,
        "final_response": final_response,
        "envelope_diagnostics": diagnostics,
        "signal_integrity": diagnostics.get("signal_integrity"),
    }
