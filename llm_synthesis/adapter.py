"""LLM adapters for synthesis generation.

Provides a base interface and concrete adapters for OpenAI-compatible
APIs and a deterministic mock for testing.
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Optional


class BaseLLMAdapter(ABC):
    """Abstract base for all LLM adapters."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send a prompt to the LLM and return the raw response text.

        Args:
            prompt: The fully formatted prompt string.

        Returns:
            Raw string response from the model (expected to be JSON).
        """


class OpenAILLMAdapter(BaseLLMAdapter):
    """Adapter for OpenAI-compatible chat completion APIs.

    Configured for deterministic, non-streaming output with
    low temperature suitable for structured JSON generation.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        max_tokens: int = 2048,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """Initialise the OpenAI adapter.

        Args:
            model: Model identifier.
            max_tokens: Maximum tokens in the completion.
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            base_url: Optional base URL for OpenAI-compatible endpoints.
        """
        try:
            from openai import OpenAI  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAILLMAdapter. "
                "Install it with: pip install openai"
            ) from exc

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        client_kwargs: dict = {"api_key": resolved_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = OpenAI(**client_kwargs)
        self._model = model
        self._max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        """Call the OpenAI chat completion API.

        Args:
            prompt: The fully formatted prompt string.

        Returns:
            Raw string content from the model response.
        """
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=1,
            max_tokens=self._max_tokens,
            stream=False,
            seed=42,
        )
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Fixed mock response used for local testing.
# ---------------------------------------------------------------------------
_MOCK_RESPONSE = {
    "insight": "Mock insight for testing purposes.",
    "evidence": "Finding A identified in test data; Finding B identified in test data.",
    "impact": "No real risk - this is a test fixture.",
    "recommended_action": "Verify integration with upstream nodes.",
    "priority": "Low",
    "confidence_score": 0.95,
}

_MOCK_RESPONSE_JSON = json.dumps(_MOCK_RESPONSE, indent=2)


class MockLLMAdapter(BaseLLMAdapter):
    """Deterministic adapter that returns a fixed valid JSON response.

    Used for local testing and CI pipelines where no LLM API
    is available.
    """

    def generate(self, prompt: str) -> str:
        """Return a fixed JSON string regardless of input.

        Args:
            prompt: Ignored - present only to satisfy the interface.

        Returns:
            A valid JSON string that can be normalized into InsightOutput.
        """
        return _MOCK_RESPONSE_JSON
