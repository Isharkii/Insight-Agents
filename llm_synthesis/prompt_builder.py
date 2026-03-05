"""Structured prompt builder for LLM synthesis."""

import json
import os
from typing import Dict

from llm_synthesis.schema import InsightOutput

_SCHEMA_JSON = json.dumps(InsightOutput.model_json_schema(), indent=2)

_EXAMPLE_OUTPUT = json.dumps(
    {
        "competitive_analysis": {
            "summary": "Peer benchmark metrics indicate competitor-led pressure on growth and retention.",
            "market_position": "The company is positioned as a challenger versus higher-share benchmark peers.",
            "relative_performance": "MRR growth trails benchmark growth while churn is above peer median.",
            "key_advantages": [
                "Stronger ARPU versus benchmark median improves monetization efficiency."
            ],
            "key_vulnerabilities": [
                "Higher churn versus competitor benchmark weakens retention durability."
            ],
            "confidence": 0.72,
        },
        "strategic_recommendations": {
            "immediate_actions": [
                "Address competitor churn gap by targeting at-risk segments where peer retention is stronger."
            ],
            "mid_term_moves": [
                "Build a roadmap to close growth-rate weakness versus benchmark peers over the next two quarters."
            ],
            "defensive_strategies": [
                "Protect accounts where competitor strength in retention metrics is highest."
            ],
            "offensive_strategies": [
                "Exploit competitor weakness in ARPU efficiency with differentiated packaging and upsell motions."
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

CONTENT RULES:
- competitive_analysis must ONLY reference competitor data and metrics.
- strategic_recommendations must explicitly reference competitor gaps, strengths, and weaknesses.
- No generic advice.
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

CONTENT RULES:
- competitive_analysis fields should analyze the entity's own performance trajectory,
  revenue trends, growth momentum, retention patterns, and risk exposure.
- Use terms like "performance", "trend", "growth", "revenue", "retention",
  "risk", "momentum", "trajectory", "decline", "volatility" in your analysis.
- strategic_recommendations must reference specific metrics, performance dimensions,
  or risk areas. Use terms like "growth", "revenue", "retention", "churn",
  "risk", "trend", "improve", "reduce", "increase", "target", "address",
  "strength", "weakness", "vulnerability", "opportunity".
- No generic advice (e.g., "improve performance", "focus on growth").
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
            "summary": "Revenue growth momentum has decelerated over the past two quarters with a declining trend in recurring revenue.",
            "market_position": "The entity shows vulnerability in retention metrics, with churn risk increasing alongside revenue volatility.",
            "relative_performance": "MRR growth rate dropped from 8% to 3% while recurring revenue trajectory signals a potential plateau.",
            "key_advantages": [
                "Revenue base remains stable despite growth deceleration, providing a foundation for recovery."
            ],
            "key_vulnerabilities": [
                "Declining growth momentum and rising churn risk expose revenue durability weakness."
            ],
            "confidence": 0.75,
        },
        "strategic_recommendations": {
            "immediate_actions": [
                "Address churn risk by targeting at-risk segments where retention metrics show the steepest decline."
            ],
            "mid_term_moves": [
                "Build a revenue growth recovery plan to improve recurring revenue trajectory over the next two quarters."
            ],
            "defensive_strategies": [
                "Reduce churn vulnerability by strengthening retention in segments showing the highest revenue decline."
            ],
            "offensive_strategies": [
                "Target revenue growth opportunity in segments where recurring revenue momentum is still positive."
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
        data_kwargs: Dict = {
            "kpi_data": kpi_data,
            "forecast_data": forecast_data,
            "risk_data": risk_data,
            "root_cause": root_cause,
            "segmentation": segmentation,
            "prioritization": prioritization,
        }
        if has_competitor_data and competitor_signals:
            data_kwargs["competitor_benchmark_signals"] = competitor_signals
        sections = self._format_data_sections(**data_kwargs)

        quality_context = self._format_quality_context(
            confidence_score, missing_signals or [],
        )

        if has_competitor_data:
            system_instructions = _SYSTEM_INSTRUCTIONS_COMPETITOR
            example_output = _EXAMPLE_OUTPUT
            task_instruction = (
                "Synthesize the provided data into a single JSON object "
                "matching the schema above. Focus strictly on competitor benchmarking "
                "content and avoid non-competitive generic recommendations."
            )
        else:
            system_instructions = _SYSTEM_INSTRUCTIONS_SELF_ANALYSIS
            example_output = _EXAMPLE_OUTPUT_SELF_ANALYSIS
            task_instruction = (
                "Synthesize the provided data into a single JSON object "
                "matching the schema above. Focus on the entity's own performance "
                "trends, strengths, weaknesses, and risk areas. Reference specific "
                "metrics and avoid generic recommendations."
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
    ) -> str:
        """Build a data-quality section for the LLM prompt."""
        quality: Dict = {
            "deterministic_confidence": round(confidence_score, 4),
            "missing_signals": missing_signals or [],
        }
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
        for key, value in data.items():
            if value in (None, {}, []):
                continue
            title = key.replace("_", " ").title()
            body = json.dumps(value, indent=2, default=str)
            if len(body) > _MAX_SECTION_CHARS:
                # Truncate overly large sections to keep token usage within limits.
                body = (
                    body[: _MAX_SECTION_CHARS]
                    + "\n... (truncated to stay under token limits)"
                )
            parts.append(_SECTION_TEMPLATE.format(title=title, data=body))
        if not parts:
            return _SECTION_TEMPLATE.format(
                title="Available Signals",
                data=json.dumps({"status": "no_signals_available"}, indent=2),
            )
        return "\n".join(parts)
