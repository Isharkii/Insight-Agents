"""Structured prompt builder for LLM synthesis."""

import json
import logging
import os
from typing import Dict

_logger = logging.getLogger(__name__)

from llm_synthesis.schema import InsightOutput

_SCHEMA_JSON = json.dumps(InsightOutput.model_json_schema(), indent=2)

_EXAMPLE_OUTPUT = json.dumps(
    {
        "competitive_analysis": {
            "summary": "MRR growth of 4.2% trails the peer median of 7.8% by 3.6pp, while churn at 5.1% exceeds the benchmark 3.2% by 1.9pp — indicating competitor-led pressure on both acquisition and retention.",
            "market_position": "Challenger: ranked 4th of 6 peers on composite growth-retention score (0.61 vs leader 0.84).",
            "relative_performance": "ARPU of $142 exceeds peer median $118 (+20%), but net revenue retention of 94% lags peer median 102%, signaling monetization strength offset by expansion weakness.",
            "key_advantages": [
                "ARPU 20% above peer median ($142 vs $118) provides pricing power for margin defense."
            ],
            "key_vulnerabilities": [
                "Churn 1.9pp above peer median erodes 2.3% of MRR monthly; at current trajectory, annual churn-driven revenue loss reaches $1.4M."
            ],
            "confidence": 0.72,
        },
        "strategic_recommendations": {
            "immediate_actions": [
                "Reduce churn from 5.1% toward peer median 3.2% by deploying retention triggers at the month-2 drop-off identified in cohort data — estimated $460K annual revenue recovery."
            ],
            "mid_term_moves": [
                "Close the 3.6pp MRR growth gap by targeting the 18% conversion-to-paid rate (vs peer 24%) with onboarding optimization over Q3-Q4."
            ],
            "defensive_strategies": [
                "Lock in top-decile accounts (ARPU > $200) with annual contracts before competitors undercut on price — these 120 accounts represent 38% of MRR."
            ],
            "offensive_strategies": [
                "Leverage ARPU advantage to fund targeted acquisition in the mid-market segment where peer churn data shows 8.4% dissatisfaction rate."
            ],
        },
    },
    indent=2,
)

_SYSTEM_INSTRUCTIONS_COMPETITOR = """\
You are a competitor benchmarking analyst.

STRICT RULES:
- Do NOT compute, calculate, or derive any new numbers.
- Use ONLY the data provided below. Do not infer beyond what is given.
- Return strictly valid JSON matching the schema defined below.
- Output exactly one JSON object with only the schema fields.
- Do NOT include any text outside the JSON object.
- Do NOT wrap the JSON in markdown code fences.
- JSON MUST follow this exact top-level shape:
  {
    "competitive_analysis": {...},
    "strategic_recommendations": {...}
  }

SIGNAL-FIRST REASONING:
- A "Signal Summary" section below provides classified signals (growth_trend,
  volatility_level, forecast_direction, cohort_health, primary_risk).
- Reason from these classified signals, NOT from raw KPI numbers.
- Do NOT repeat raw metric values. Instead, reference the signal classification
  and competitor deltas.
- Structure your analysis as:
  1. Key Signal Summary — what the signals tell us about competitive position
  2. Primary Drivers — what is driving competitive gaps
  3. Risk Assessment — what competitive risks the signals expose
  4. Strategic Recommendations — what actions to take based on signals

SPECIFICITY RULES (CRITICAL):
- Every claim MUST cite a specific number, percentage, or delta from the provided data.
- Every recommendation MUST include a quantified impact estimate or measurable target.
- BANNED PHRASES (your output will be rejected if these appear):
  "focus on improving", "consider implementing", "explore opportunities",
  "optimize performance", "enhance capabilities", "leverage strengths",
  "address weaknesses", "drive growth", "stabilize revenue streams",
  "build a roadmap", "strengthen retention", "improve metrics".
- Instead of "address churn risk", write "reduce churn from 5.1% toward 3.2%
  by targeting month-2 drop-off in Q3 cohorts — estimated $460K recovery".
- Reference specific cohorts, time periods, segments, or metric values.
- If the data contains growth rates, slopes, or deltas, cite them directly.

CONTENT RULES:
- competitive_analysis must ONLY reference competitor data and metrics.
- strategic_recommendations must explicitly reference competitor gaps, strengths, and weaknesses.
- No generic advice. Every sentence must contain at least one specific data point.
- No repetition across recommendation sections.
- Tone must match deterministic confidence.
- If confidence < 0.5, every recommendation must be labeled as "Conditional:".

CONFIDENCE-AWARE NARRATIVE RULES:
- A "Data Quality" section below reports the deterministic confidence score
  (0.0 to 1.0) and any missing signals.
- If confidence >= 0.8: use definitive language ("revenue declined 12%").
- If 0.6 <= confidence < 0.8: use cautious language ("revenue appears to
  have declined", "available data suggests").
- If 0.4 <= confidence < 0.6: use hedged language ("limited data indicates",
  "with significant uncertainty").  Explicitly note data gaps in the insight.
- Your competitive_analysis.confidence field MUST NOT exceed the deterministic confidence
  value provided below.
"""

_SYSTEM_INSTRUCTIONS_SELF_ANALYSIS = """\
You are a business performance analyst.

STRICT RULES:
- Do NOT compute, calculate, or derive any new numbers.
- Use ONLY the data provided below. Do not infer beyond what is given.
- Return strictly valid JSON matching the schema defined below.
- Output exactly one JSON object with only the schema fields.
- Do NOT include any text outside the JSON object.
- Do NOT wrap the JSON in markdown code fences.
- JSON MUST follow this exact top-level shape:
  {
    "competitive_analysis": {...},
    "strategic_recommendations": {...}
  }

CONTEXT: No competitor or peer benchmark data is available.
Analyze the entity's OWN performance trends, strengths, and weaknesses.

SIGNAL-FIRST REASONING:
- A "Signal Summary" section below provides classified signals (growth_trend,
  volatility_level, forecast_direction, cohort_health, primary_risk).
- Reason from these classified signals, NOT from raw KPI numbers.
- Do NOT repeat raw metric values. Instead, reference the signal classification
  (e.g., "growth is accelerating" not "revenue was 10000 then 12000").
- Structure your analysis as:
  1. Key Signal Summary — what the signals tell us
  2. Primary Drivers — what is driving the observed trends
  3. Risk Assessment — what risks the signals expose
  4. Strategic Recommendations — what actions to take based on signals

SPECIFICITY RULES (CRITICAL):
- Every claim MUST cite a specific number, percentage, or delta from the provided data.
- Every recommendation MUST include a quantified impact estimate or measurable target.
- BANNED PHRASES (your output will be rejected if these appear):
  "focus on improving", "consider implementing", "explore opportunities",
  "optimize performance", "enhance capabilities", "leverage strengths",
  "address weaknesses", "drive growth", "stabilize revenue streams",
  "build a roadmap", "strengthen retention", "improve metrics",
  "focus on stabilizing", "improve performance", "focus on growth".
- Instead of "declining growth trend — focus on stabilizing revenue", write
  "revenue declined 8.3% QoQ ($242K to $222K) — deploy retention triggers
  targeting the month-2 churn spike (4.7% vs 1.9% baseline) to recover $89K annually".
- Reference specific cohorts, time periods, growth rates, slopes, or metric values.
- If the data contains forecast slopes, deviation percentages, churn deltas,
  or growth horizons, cite them directly with their numeric values.

CONTENT RULES:
- competitive_analysis fields should analyze the entity's own performance trajectory
  with specific numbers: revenue values, growth rates, churn percentages, slopes.
- strategic_recommendations must reference specific metrics with their values
  and include quantified targets or impact estimates.
- No generic advice. Every sentence must contain at least one specific data point.
- No repetition across recommendation sections.
- Tone must match deterministic confidence.
- If confidence < 0.5, every recommendation must be labeled as "Conditional:".

CONFIDENCE-AWARE NARRATIVE RULES:
- A "Data Quality" section below reports the deterministic confidence score
  (0.0 to 1.0) and any missing signals.
- If confidence >= 0.8: use definitive language ("revenue declined 12%").
- If 0.6 <= confidence < 0.8: use cautious language ("revenue appears to
  have declined", "available data suggests").
- If 0.4 <= confidence < 0.6: use hedged language ("limited data indicates",
  "with significant uncertainty").  Explicitly note data gaps in the insight.
- Your competitive_analysis.confidence field MUST NOT exceed the deterministic confidence
  value provided below.
"""

_EXAMPLE_OUTPUT_SELF_ANALYSIS = json.dumps(
    {
        "competitive_analysis": {
            "summary": "Revenue declined 8.3% QoQ ($242K to $222K) driven by churn acceleration from 3.1% to 4.7% — cohorts acquired in Q3 show 2.4x higher month-2 drop-off than Q1 cohorts.",
            "market_position": "Declining trajectory with high-severity risk: forecast slope is -0.034 with deviation 12.6%, signaling continued contraction without intervention.",
            "relative_performance": "Short-term growth at -8.3% vs mid-term -4.1% shows accelerating decline (trend acceleration -0.042). LTV/CAC ratio degraded from 3.2 to 2.6 over the period.",
            "key_advantages": [
                "Core customer base (cohorts Q1-Q2) retains at 92% month-over-month — decline is concentrated in recent acquisition cohorts, not the installed base."
            ],
            "key_vulnerabilities": [
                "Q3 cohort churn acceleration of 12% QoQ will compound to $89K additional annual revenue loss if month-2 retention is not addressed within 60 days."
            ],
            "confidence": 0.75,
        },
        "strategic_recommendations": {
            "immediate_actions": [
                "Deploy retention intervention targeting month-2 churn spike in Q3 cohorts (4.7% vs 1.9% baseline) — onboarding sequence redesign with success milestones by day 14 and day 30."
            ],
            "mid_term_moves": [
                "Reverse the -0.034 forecast slope by shifting acquisition spend toward channels that produced Q1-Q2 cohort quality (92% retention) vs Q3 channels (78% retention)."
            ],
            "defensive_strategies": [
                "Protect the $178K MRR from pre-Q3 cohorts with proactive health scoring — flag accounts with usage decline >15% for CSM outreach before renewal."
            ],
            "offensive_strategies": [
                "LTV/CAC of 2.6 still supports expansion: target upsell to the 34 accounts with usage above 80th percentile but on base-tier plans — estimated $12K incremental MRR."
            ],
        },
    },
    indent=2,
)

_SECTION_TEMPLATE = """\
## {title}
```json
{data}
```
"""

_MAX_SECTION_CHARS = int(os.getenv("PROMPT_SECTION_CHAR_LIMIT", "6000"))


class SynthesisPromptBuilder:
    """Builds a deterministic structured prompt for LLM synthesis.

    Combines upstream node outputs into a single prompt that
    instructs the LLM to synthesize insights into a structured
    JSON response matching InsightOutput schema.
    """

    def build_prompt(
        self,
        kpi_data: Dict,
        forecast_data: Dict,
        risk_data: Dict,
        root_cause: Dict,
        segmentation: Dict,
        prioritization: Dict,
        confidence_score: float = 1.0,
        missing_signals: list | None = None,
        has_competitor_data: bool = False,
        competitor_signals: Dict | None = None,
        conflict_metadata: Dict | None = None,
        signal_enrichment: Dict | None = None,
        isolated_layers: list | None = None,
        degraded_layers: list | None = None,
        reasoning_warnings: list | None = None,
    ) -> str:
        """Build the full synthesis prompt from upstream data.

        Args:
            kpi_data: KPI metrics from the insight layer.
            forecast_data: Forecast projections.
            risk_data: Risk assessment results.
            root_cause: Root cause analysis output.
            segmentation: Segmentation / cohort analysis output.
            prioritization: Prioritized action items.
            confidence_score: Deterministic confidence from signal quality.
            missing_signals: Names of signals that are unavailable.
            has_competitor_data: Whether competitor/peer data is available.
                When False, prompts shift to self-analysis mode.
            competitor_signals: Numeric-only competitive benchmark signals.
                Only included when has_competitor_data is True.

        Returns:
            A fully formatted prompt string ready for LLM consumption.
        """
        data_kwargs: Dict = {}
        # ── Insight Digest (Decision-Grade) ──
        # When the enrichment node produces an insight_digest, use it as
        # the PRIMARY data source — it's compressed, structured, and
        # LLM-optimized.  Raw payloads are still included as fallback
        # but the digest takes visual precedence in the prompt.
        insight_digest = None
        if signal_enrichment:
            insight_digest = signal_enrichment.get("insight_digest")
            key_metrics = signal_enrichment.get("key_metrics")
            if isinstance(key_metrics, dict) and key_metrics:
                data_kwargs["key_metrics_reference"] = key_metrics

        if isinstance(insight_digest, dict) and insight_digest:
            # Structured digest replaces raw blobs — no truncation risk
            data_kwargs["insight_digest"] = insight_digest
            # Include signal summary (lightweight classified signals)
            if signal_enrichment:
                # Strip the digest and key_metrics to avoid duplication
                summary = {
                    k: v for k, v in signal_enrichment.items()
                    if k not in ("insight_digest", "key_metrics")
                }
                if summary:
                    data_kwargs["signal_summary"] = summary
        else:
            # Fallback: legacy mode — raw payloads
            if signal_enrichment:
                data_kwargs["signal_summary"] = signal_enrichment

        data_kwargs.update({
            "kpi_data": kpi_data,
            "forecast_data": forecast_data,
            "risk_data": risk_data,
            "root_cause": root_cause,
            "segmentation": segmentation,
            "prioritization": prioritization,
        })
        if conflict_metadata:
            data_kwargs["signal_conflicts"] = conflict_metadata
        if has_competitor_data and competitor_signals:
            data_kwargs["competitor_benchmark_signals"] = competitor_signals
        sections = self._format_data_sections(**data_kwargs)

        quality_context = self._format_quality_context(
            confidence_score,
            missing_signals or [],
            isolated_layers=isolated_layers,
            degraded_layers=degraded_layers,
            reasoning_warnings=reasoning_warnings,
        )

        if has_competitor_data:
            system_instructions = _SYSTEM_INSTRUCTIONS_COMPETITOR
            example_output = _EXAMPLE_OUTPUT
            task_instruction = (
                "Synthesize the provided data into a single JSON object "
                "matching the schema above. Focus strictly on competitor benchmarking "
                "with SPECIFIC numbers: cite growth deltas, churn gaps, ARPU "
                "differences, and rank positions from the data. Every recommendation "
                "must include a quantified target or impact. Generic advice will be rejected."
            )
        else:
            system_instructions = _SYSTEM_INSTRUCTIONS_SELF_ANALYSIS
            example_output = _EXAMPLE_OUTPUT_SELF_ANALYSIS
            task_instruction = (
                "Synthesize the provided data into a single JSON object "
                "matching the schema above. Focus on the entity's own performance "
                "with SPECIFIC numbers from the data: cite growth rates, revenue "
                "values, churn percentages, forecast slopes, and cohort metrics. "
                "Every recommendation must include a quantified target or impact. "
                "Generic advice without numbers will be rejected."
            )

        return (
            f"{system_instructions}\n"
            f"# PROVIDED DATA\n\n{sections}\n"
            f"{quality_context}"
            f"# OUTPUT SCHEMA\n\n"
            f"Your response MUST conform to this JSON schema:\n\n"
            f"```json\n{_SCHEMA_JSON}\n```\n\n"
            f"# EXAMPLE OUTPUT\n\n"
            f"```json\n{example_output}\n```\n\n"
            f"# TASK\n\n"
            f"{task_instruction}"
        )

    @staticmethod
    def _format_quality_context(
        confidence_score: float,
        missing_signals: list[str],
        isolated_layers: list[str] | None = None,
        degraded_layers: list[str] | None = None,
        reasoning_warnings: list[str] | None = None,
    ) -> str:
        """Build a data-quality section for the LLM prompt.

        Includes signal integrity diagnostics so the LLM knows which
        data sources are unavailable, isolated, or degraded.
        """
        quality: Dict = {
            "deterministic_confidence": round(confidence_score, 4),
            "missing_signals": missing_signals or [],
        }
        if isolated_layers:
            quality["isolated_signals"] = isolated_layers
            quality["isolation_note"] = (
                "These signals were excluded from scoring due to "
                "low quality (e.g. R² < 0.2, broken modules). "
                "Do NOT reference forecast data if forecast is isolated."
            )
        if degraded_layers:
            quality["degraded_signals"] = degraded_layers
            quality["degradation_note"] = (
                "These signals have reduced reliability. "
                "Cite them with appropriate hedging language."
            )
        if reasoning_warnings:
            quality["reasoning_warnings"] = reasoning_warnings
        if confidence_score >= 0.8:
            quality["tone_directive"] = "definitive"
        elif confidence_score >= 0.6:
            quality["tone_directive"] = "cautious"
        else:
            quality["tone_directive"] = "hedged"
        body = json.dumps(quality, indent=2)
        return f"## Data Quality\n```json\n{body}\n```\n\n"

    def _format_data_sections(self, **data: Dict) -> str:
        """Format each data dict as a labeled JSON section.

        Args:
            **data: Named data dictionaries to include in the prompt.

        Returns:
            Concatenated formatted sections.
        """
        parts = []
        truncated_sections: list[str] = []
        for key, value in data.items():
            if value in (None, {}, []):
                continue
            title = key.replace("_", " ").title()
            body = json.dumps(value, indent=2, default=str)
            if len(body) > _MAX_SECTION_CHARS:
                original_len = len(body)
                body = (
                    body[: _MAX_SECTION_CHARS]
                    + f"\n... [TRUNCATED: {original_len} → {_MAX_SECTION_CHARS} chars. "
                    f"Some data in '{title}' section is missing from this prompt.]"
                )
                truncated_sections.append(
                    f"{title} ({original_len} → {_MAX_SECTION_CHARS} chars)"
                )
                _logger.warning(
                    "Prompt section '%s' truncated: %d → %d chars",
                    title, original_len, _MAX_SECTION_CHARS,
                )
            parts.append(_SECTION_TEMPLATE.format(title=title, data=body))
        if truncated_sections:
            warning = (
                "**WARNING — Data Truncation**\n"
                "The following sections were truncated to fit token limits. "
                "Some metrics may be absent from the data below. "
                "Do not infer values for missing data.\n"
                + "\n".join(f"- {s}" for s in truncated_sections)
                + "\n\n"
            )
            parts.insert(0, warning)
        if not parts:
            return _SECTION_TEMPLATE.format(
                title="Available Signals",
                data=json.dumps({"status": "no_signals_available"}, indent=2),
            )
        return "\n".join(parts)
