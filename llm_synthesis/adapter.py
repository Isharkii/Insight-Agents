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
        model: str = "gpt-5.4",
        max_tokens: int = 2048,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialise the OpenAI adapter.

        Args:
            model: Model identifier.
            max_tokens: Maximum tokens in the completion.
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            base_url: Optional base URL for OpenAI-compatible endpoints.
            timeout: Request timeout in seconds (default 45s).
        """
        try:
            from openai import OpenAI  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAILLMAdapter. "
                "Install it with: pip install openai"
            ) from exc

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        client_kwargs: dict = {"api_key": resolved_key, "timeout": timeout}
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = OpenAI(**client_kwargs)
        self._model = str(model or "").strip() or "gpt-5.4"
        self._max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        """Call the OpenAI chat completion API.

        Args:
            prompt: The fully formatted prompt string.

        Returns:
            Raw string content from the model response.
        """
        base_kwargs = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "top_p": 1,
            "stream": False,
            "seed": 42,
        }
        try:
            # GPT-5.x chat models require max_completion_tokens.
            response = self._client.chat.completions.create(
                **base_kwargs,
                max_completion_tokens=self._max_tokens,
            )
        except TypeError:
            # Backward compatibility for older SDKs/models.
            response = self._client.chat.completions.create(
                **base_kwargs,
                max_tokens=self._max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            message = str(exc).lower()
            if "unsupported parameter" in message and "max_completion_tokens" in message:
                response = self._client.chat.completions.create(
                    **base_kwargs,
                    max_tokens=self._max_tokens,
                )
            else:
                raise
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Fixed mock responses used for local testing.
# ---------------------------------------------------------------------------
_MOCK_RESPONSE_COMPETITOR = {
    "competitive_analysis": {
        "summary": "Mock competitor benchmark summary based on provided peer metrics.",
        "market_position": "Mock market position indicates balanced peer standing.",
        "relative_performance": "Mock relative performance shows minor benchmark gaps in growth and churn metrics.",
        "key_advantages": ["Mock advantage: stronger peer-comparative ARPU metric."],
        "key_vulnerabilities": ["Mock vulnerability: weaker peer-comparative churn metric."],
        "confidence": 0.95,
    },
    "strategic_recommendations": {
        "immediate_actions": [
            "Address competitor gap in churn metric by targeting accounts where peer strength is highest."
        ],
        "mid_term_moves": [
            "Close growth weakness versus competitor benchmark through segment-specific pricing tests."
        ],
        "defensive_strategies": [
            "Defend segments where competitor strength in retention metrics is increasing."
        ],
        "offensive_strategies": [
            "Exploit competitor weakness in ARPU efficiency with focused upsell motions."
        ],
    },
}

_MOCK_RESPONSE_SELF_ANALYSIS = {
    "competitive_analysis": {
        "summary": "Mock performance trend analysis shows revenue growth momentum decelerating over the observed period.",
        "market_position": "Mock trajectory indicates a risk of revenue plateau with declining growth rate metrics.",
        "relative_performance": "Mock recurring revenue growth dropped while churn risk volatility increased.",
        "key_advantages": ["Mock advantage: revenue base remains stable despite growth deceleration."],
        "key_vulnerabilities": ["Mock vulnerability: declining growth momentum exposes retention risk."],
        "confidence": 0.95,
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
}

_MOCK_RESPONSE_COMPETITOR_JSON = json.dumps(_MOCK_RESPONSE_COMPETITOR, indent=2)
_MOCK_RESPONSE_SELF_ANALYSIS_JSON = json.dumps(_MOCK_RESPONSE_SELF_ANALYSIS, indent=2)


class MockLLMAdapter(BaseLLMAdapter):
    """Deterministic adapter that returns a fixed valid JSON response.

    Used for local testing and CI pipelines where no LLM API
    is available.
    """

    def generate(self, prompt: str) -> str:
        """Return a fixed JSON string matching the current analysis mode.

        Detects self-analysis vs competitor mode from the prompt content
        and returns the appropriate mock response.

        Args:
            prompt: The formatted prompt (inspected for mode detection).

        Returns:
            A valid JSON string that can be normalized into InsightOutput.
        """
        if "No competitor or peer benchmark data" in prompt:
            return _MOCK_RESPONSE_SELF_ANALYSIS_JSON
        return _MOCK_RESPONSE_COMPETITOR_JSON
