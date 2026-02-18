"""Validation layer for raw LLM synthesis output.

Parses and validates JSON strings against the InsightOutput schema.
"""

import json
import re
from typing import Any, Dict, List

from pydantic import ValidationError

from llm_synthesis.schema import InsightOutput

_REQUIRED_KEYS = (
    "insight",
    "evidence",
    "impact",
    "recommended_action",
    "priority",
    "confidence_score",
)


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


def _compress_to_insight_output(data: Dict[str, Any]) -> Dict[str, Any]:
    """Project payloads into the InsightOutput keys only."""
    return {key: data.get(key) for key in _REQUIRED_KEYS}


def validate_llm_output(raw_response: str) -> InsightOutput:
    """Parse and validate a raw LLM response string.

    Steps:
        1. Strip optional markdown fences.
        2. Parse as JSON.
        3. Project payload into InsightOutput keys.
        4. Validate against InsightOutput Pydantic model.

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

    # Step 3: Schema validation
    try:
        normalized = _compress_to_insight_output(data)
        parsed = InsightOutput.model_validate(normalized)
        return InsightOutput.model_validate({k: getattr(parsed, k) for k in _REQUIRED_KEYS})
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
