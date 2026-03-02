"""Structured prompt builder for LLM synthesis."""

import json
from typing import Dict

from llm_synthesis.schema import InsightOutput

_SCHEMA_JSON = json.dumps(InsightOutput.model_json_schema(), indent=2)

_EXAMPLE_OUTPUT = json.dumps(
    {
        "insight": "Revenue declined 12% QoQ driven by churn in the mid-market segment.",
        "evidence": "Mid-market churn rose from 4% to 9%; CAC increased 18% with no corresponding LTV gain.",
        "impact": "If unaddressed, ARR pressure is likely to continue over upcoming quarters.",
        "recommended_action": "Launch a targeted retention plan for at-risk mid-market accounts.",
        "priority": "critical",
        "confidence_score": 0.85,
        "pipeline_status": "partial",
    },
    indent=2,
)

_SYSTEM_INSTRUCTIONS = """\
You are a strategic business analyst.

STRICT RULES:
- Do NOT compute, calculate, or derive any new numbers.
- Use ONLY the data provided below. Do not infer beyond what is given.
- Return strictly valid JSON matching the schema defined below.
- Output exactly one JSON object with only the schema fields.
- Do NOT include any text outside the JSON object.
- Do NOT wrap the JSON in markdown code fences.

CONFIDENCE-AWARE NARRATIVE RULES:
- A "Data Quality" section below reports the deterministic confidence score
  (0.0 to 1.0) and any missing signals.
- If confidence >= 0.8: use definitive language ("revenue declined 12%").
- If 0.6 <= confidence < 0.8: use cautious language ("revenue appears to
  have declined", "available data suggests").
- If 0.4 <= confidence < 0.6: use hedged language ("limited data indicates",
  "with significant uncertainty").  Explicitly note data gaps in the insight.
- Your confidence_score field MUST NOT exceed the deterministic confidence
  value provided below.
"""

_SECTION_TEMPLATE = """\
## {title}
```json
{data}
```
"""


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

        Returns:
            A fully formatted prompt string ready for LLM consumption.
        """
        sections = self._format_data_sections(
            kpi_data=kpi_data,
            forecast_data=forecast_data,
            risk_data=risk_data,
            root_cause=root_cause,
            segmentation=segmentation,
            prioritization=prioritization,
        )

        quality_context = self._format_quality_context(
            confidence_score, missing_signals or [],
        )

        return (
            f"{_SYSTEM_INSTRUCTIONS}\n"
            f"# PROVIDED DATA\n\n{sections}\n"
            f"{quality_context}"
            f"# OUTPUT SCHEMA\n\n"
            f"Your response MUST conform to this JSON schema:\n\n"
            f"```json\n{_SCHEMA_JSON}\n```\n\n"
            f"# EXAMPLE OUTPUT\n\n"
            f"```json\n{_EXAMPLE_OUTPUT}\n```\n\n"
            f"# TASK\n\n"
            f"Synthesize the provided data into a single JSON object "
            f"matching the schema above. Do not compute. Use only provided data."
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
            parts.append(_SECTION_TEMPLATE.format(title=title, data=body))
        if not parts:
            return _SECTION_TEMPLATE.format(
                title="Available Signals",
                data=json.dumps({"status": "no_signals_available"}, indent=2),
            )
        return "\n".join(parts)
