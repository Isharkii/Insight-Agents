"""Structured prompt builder for LLM synthesis."""

import json
from typing import Dict

from llm_synthesis.schema import SynthesisOutput

_SCHEMA_JSON = json.dumps(SynthesisOutput.model_json_schema(), indent=2)

_EXAMPLE_OUTPUT = json.dumps(
    {
        "executive_summary": "Revenue declined 12% QoQ driven by churn in mid-market segment.",
        "key_findings": [
            "Mid-market churn rose from 4% to 9%",
            "CAC increased 18% with no corresponding LTV gain",
            "Forecast projects continued decline without intervention",
        ],
        "primary_risk": "Accelerating mid-market churn threatening ARR base.",
        "recommended_actions": [
            "Launch targeted retention campaign for mid-market accounts",
            "Review pricing structure for mid-market tier",
            "Investigate onboarding drop-off points",
        ],
        "priority_level": "Critical",
        "confidence_score": 0.85,
    },
    indent=2,
)

_SYSTEM_INSTRUCTIONS = """\
You are a strategic business analyst.

STRICT RULES:
- Do NOT compute, calculate, or derive any new numbers.
- Use ONLY the data provided below. Do not infer beyond what is given.
- Return strictly valid JSON matching the schema defined below.
- Do NOT include any text outside the JSON object.
- Do NOT wrap the JSON in markdown code fences.
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
    instructs the LLM to synthesize findings into a structured
    JSON response matching SynthesisOutput schema.
    """

    def build_prompt(
        self,
        kpi_data: Dict,
        forecast_data: Dict,
        risk_data: Dict,
        root_cause: Dict,
        segmentation: Dict,
        prioritization: Dict,
    ) -> str:
        """Build the full synthesis prompt from upstream data.

        Args:
            kpi_data: KPI metrics from the insight layer.
            forecast_data: Forecast projections.
            risk_data: Risk assessment results.
            root_cause: Root cause analysis output.
            segmentation: Segmentation / cohort analysis output.
            prioritization: Prioritized action items.

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

        return (
            f"{_SYSTEM_INSTRUCTIONS}\n"
            f"# PROVIDED DATA\n\n{sections}\n"
            f"# OUTPUT SCHEMA\n\n"
            f"Your response MUST conform to this JSON schema:\n\n"
            f"```json\n{_SCHEMA_JSON}\n```\n\n"
            f"# EXAMPLE OUTPUT\n\n"
            f"```json\n{_EXAMPLE_OUTPUT}\n```\n\n"
            f"# TASK\n\n"
            f"Synthesize the provided data into a single JSON object "
            f"matching the schema above. Do not compute. Use only provided data."
        )

    def _format_data_sections(self, **data: Dict) -> str:
        """Format each data dict as a labeled JSON section.

        Args:
            **data: Named data dictionaries to include in the prompt.

        Returns:
            Concatenated formatted sections.
        """
        parts = []
        for key, value in data.items():
            title = key.replace("_", " ").title()
            body = json.dumps(value, indent=2, default=str)
            parts.append(_SECTION_TEMPLATE.format(title=title, data=body))
        return "\n".join(parts)
