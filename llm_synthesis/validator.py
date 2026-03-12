"""Validation layer for raw LLM synthesis output.

Parses and validates JSON strings against the InsightOutput schema.
"""

import json
import re
from typing import Any, List

from pydantic import ValidationError

from llm_synthesis.schema import InsightOutput


class LLMOutputValidationError(Exception):
    """Raised when LLM output fails parsing or schema validation.

    Attributes:
        stage: Which validation step failed ("json_parse" or "schema").
        errors: List of human-readable error descriptions.
        raw_response: The original string that failed validation.
    """

    def __init__(
        self,
        stage: str,
        errors: List[str],
        raw_response: str,
    ) -> None:
        self.stage = stage
        self.errors = errors
        self.raw_response = raw_response
        message = (
            f"LLM output validation failed at stage '{stage}': "
            + "; ".join(errors)
        )
        super().__init__(message)


def _strip_markdown_fences(text: str) -> str:
    """Remove optional markdown code fences wrapping JSON.

    LLMs sometimes wrap output in ```json ... ``` despite instructions.
    This strips that wrapper so the inner JSON can be parsed.

    Args:
        text: Raw LLM response string.

    Returns:
        The text with leading/trailing code fences removed, if present.
    """
    stripped = text.strip()
    match = re.match(
        r"^```(?:json)?\s*\n?(.*?)\n?\s*```$",
        stripped,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return stripped


_LOW_CONFIDENCE_TONE_TERMS = (
    "conditional",
    "suggest",
    "appears",
    "may",
    "might",
    "could",
    "uncertain",
    "likely",
)

_RECOMMENDATION_FIELDS = (
    "immediate_actions",
    "mid_term_moves",
    "defensive_strategies",
    "offensive_strategies",
)


def _apply_conditional_labels(data: dict) -> dict:
    """Deterministically enforce confidence-governed labeling.

    When confidence < 0.5, prefixes all recommendations with "Conditional:"
    and injects cautious tone into analysis text if missing. This is a
    deterministic post-processing step — the LLM should not need to know
    about internal labeling conventions.
    """
    ca = data.get("competitive_analysis")
    sr = data.get("strategic_recommendations")
    if not isinstance(ca, dict) or not isinstance(sr, dict):
        return data

    confidence = ca.get("confidence")
    if confidence is None or confidence >= 0.5:
        return data

    # Deep-copy the mutable parts we'll modify
    ca = dict(ca)
    sr = dict(sr)

    # Prefix recommendations with "Conditional:" if not already
    for field in _RECOMMENDATION_FIELDS:
        items = sr.get(field)
        if isinstance(items, list):
            sr[field] = [
                item if item.strip().lower().startswith("conditional:")
                else f"Conditional: {item}"
                for item in items
            ]

    # Inject cautious tone into analysis summary if missing
    analysis_text = " ".join(
        str(ca.get(f, "")) for f in ("summary", "market_position", "relative_performance")
    ).lower()
    if not any(term in analysis_text for term in _LOW_CONFIDENCE_TONE_TERMS):
        summary = ca.get("summary", "")
        ca["summary"] = f"Conditional: {summary}" if summary else "Conditional: analysis uncertain."

    data = dict(data)
    data["competitive_analysis"] = ca
    data["strategic_recommendations"] = sr
    return data


def validate_llm_output(raw_response: str) -> InsightOutput:
    """Parse and validate a raw LLM response string.

    Steps:
        1. Strip optional markdown fences.
        2. Parse as JSON.
        3. Validate against InsightOutput Pydantic model.

    Args:
        raw_response: The raw string returned by the LLM adapter.

    Returns:
        A validated InsightOutput instance.

    Raises:
        LLMOutputValidationError: If JSON parsing or schema validation fails.
    """
    cleaned = _strip_markdown_fences(raw_response)

    # Step 1: JSON parse
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError) as exc:
        raise LLMOutputValidationError(
            stage="json_parse",
            errors=[str(exc)],
            raw_response=raw_response,
        ) from exc

    # Step 2: Object shape
    if not isinstance(data, dict):
        raise LLMOutputValidationError(
            stage="schema",
            errors=["top-level JSON must be an object"],
            raw_response=raw_response,
        )

    # Step 2.5: Deterministic post-processing — apply confidence-governed
    # tone rules *before* schema validation so the LLM doesn't need to know
    # about internal labeling conventions.
    data = _apply_conditional_labels(data)

    # Step 3: Schema validation (rejects unknown fields via model config)
    try:
        return InsightOutput.model_validate(data)
    except ValidationError as exc:
        errors = [
            f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}"
            for e in exc.errors()
        ]
        raise LLMOutputValidationError(
            stage="schema",
            errors=errors,
            raw_response=raw_response,
        ) from exc
