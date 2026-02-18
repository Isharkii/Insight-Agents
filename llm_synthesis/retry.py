"""Retry logic for LLM formatting errors.

Retries only on JSON parse or schema validation failures.
Does NOT retry on business-logic issues or adapter transport errors.
"""

import logging
from typing import List

from llm_synthesis.adapter import BaseLLMAdapter
from llm_synthesis.schema import InsightOutput
from llm_synthesis.validator import LLMOutputValidationError, validate_llm_output

logger = logging.getLogger(__name__)

_RETRYABLE_STAGES = frozenset({"json_parse", "schema"})


class LLMRetryExhaustedError(Exception):
    """Raised when all retry attempts fail validation.

    Attributes:
        attempts: Total number of attempts made (initial + retries).
        last_error: The validation error from the final attempt.
        history: Validation errors from every failed attempt.
    """

    def __init__(
        self,
        attempts: int,
        last_error: LLMOutputValidationError,
        history: List[LLMOutputValidationError],
    ) -> None:
        self.attempts = attempts
        self.last_error = last_error
        self.history = history
        super().__init__(
            f"LLM output validation failed after {attempts} attempt(s). "
            f"Last error: {last_error}"
        )


def generate_with_retry(
    adapter: BaseLLMAdapter,
    prompt: str,
    max_retries: int = 2,
) -> InsightOutput:
    """Generate LLM output with retry on formatting errors.

    Calls ``adapter.generate()`` and validates the response. If validation
    fails with a JSON parse or schema error, retries up to ``max_retries``
    additional times. Non-retryable errors are raised immediately.

    Args:
        adapter: An LLM adapter implementing ``generate(prompt) -> str``.
        prompt: The fully formatted prompt string.
        max_retries: Maximum number of *additional* attempts after the
            first failure. Total attempts = 1 + max_retries.

    Returns:
        A validated ``InsightOutput`` instance.

    Raises:
        LLMOutputValidationError: If a non-retryable validation error occurs.
        LLMRetryExhaustedError: If all attempts fail with retryable errors.
    """
    errors: List[LLMOutputValidationError] = []
    total_attempts = 1 + max_retries

    for attempt in range(1, total_attempts + 1):
        raw = adapter.generate(prompt)

        try:
            result = validate_llm_output(raw)
            if attempt > 1:
                logger.info(
                    "LLM output validated on attempt %d/%d",
                    attempt,
                    total_attempts,
                )
            return result

        except LLMOutputValidationError as exc:
            if exc.stage not in _RETRYABLE_STAGES:
                raise

            errors.append(exc)
            logger.warning(
                "Attempt %d/%d failed at stage '%s': %s",
                attempt,
                total_attempts,
                exc.stage,
                "; ".join(exc.errors),
            )

    raise LLMRetryExhaustedError(
        attempts=total_attempts,
        last_error=errors[-1],
        history=errors,
    )
