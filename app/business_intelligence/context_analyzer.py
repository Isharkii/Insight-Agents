"""Extract structured business context from free-text descriptions.

Uses an LLM to classify industry, business model, target market, and
generate contextual search intents — but never to compute or infer
financial metrics.  Output is strictly validated against a Pydantic
schema before being returned.
"""

from __future__ import annotations

import json
import logging
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from llm_synthesis.adapter import BaseLLMAdapter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------

_VALID_BUSINESS_MODELS = frozenset({
    "saas",
    "marketplace",
    "ecommerce",
    "agency",
    "fintech",
    "hardware",
    "subscription",
    "freemium",
    "consulting",
    "platform",
    "other",
})


class BusinessContext(BaseModel):
    """Structured business context extracted from a free-text description.

    This schema captures *classification* data only.  It must never contain
    computed financial metrics — those belong in the deterministic KPI layer.
    """

    model_config = ConfigDict(extra="forbid")

    industry: str = Field(
        min_length=2,
        max_length=100,
        description="Primary industry vertical (e.g. 'healthcare', 'fintech').",
    )
    business_model: str = Field(
        min_length=2,
        max_length=50,
        description="Core business model type.",
    )
    target_market: str = Field(
        min_length=2,
        max_length=200,
        description=(
            "Geographic and demographic target market "
            "(e.g. 'SMB healthcare clinics in India')."
        ),
    )
    macro_dependencies: List[str] = Field(
        min_length=1,
        max_length=10,
        description=(
            "External macro factors this business is sensitive to "
            "(e.g. 'healthcare regulation', 'USD/INR exchange rate')."
        ),
    )
    search_intents: List[str] = Field(
        min_length=5,
        max_length=5,
        description=(
            "Exactly 5 web-search queries an analyst would run "
            "to benchmark this business against competitors."
        ),
    )
    risk_factors: List[str] = Field(
        min_length=1,
        max_length=10,
        description=(
            "Key business risks inferred from the description "
            "(e.g. 'regulatory dependency', 'single-market concentration')."
        ),
    )

    @field_validator("business_model")
    @classmethod
    def _normalise_business_model(cls, v: str) -> str:
        lowered = v.strip().lower().replace(" ", "_").replace("-", "_")
        if lowered not in _VALID_BUSINESS_MODELS:
            logger.warning(
                "Business model '%s' not in canonical set; accepting as-is.",
                lowered,
            )
        return lowered


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a business analyst that extracts structured context from a \
free-text business description.

RULES — read carefully:
1. Return ONLY valid JSON matching the schema below.  No markdown fences.
2. Do NOT invent, estimate, or hallucinate any financial metrics, revenue \
   figures, growth rates, or KPIs.  You are classifying — not computing.
3. ``search_intents`` must contain exactly 5 queries an analyst would use \
   to find competitor benchmarks, market sizing, or regulatory landscape \
   for this specific business.  Make them specific, not generic.
4. ``macro_dependencies`` should list real external factors (regulation, \
   FX rates, commodity prices, policy changes) — not internal business \
   concerns.
5. ``risk_factors`` should be concrete and specific to the described \
   business, not boilerplate.
6. ``business_model`` must be one of: saas, marketplace, ecommerce, \
   agency, fintech, hardware, subscription, freemium, consulting, \
   platform, other.

JSON schema:
{
  "industry": "<string>",
  "business_model": "<string>",
  "target_market": "<string>",
  "macro_dependencies": ["<string>", ...],
  "search_intents": ["<string>", "<string>", "<string>", "<string>", "<string>"],
  "risk_factors": ["<string>", ...]
}
"""

_USER_TEMPLATE = "Business description:\n{description}"


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ContextAnalyzerError(Exception):
    """Raised when context extraction fails after all attempts."""

    def __init__(self, attempts: int, last_error: str) -> None:
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"Context extraction failed after {attempts} attempt(s): {last_error}"
        )


class ContextAnalyzer:
    """Extract structured business context via an LLM adapter.

    The adapter is injected at construction time so the caller decides
    which model backend to use (OpenAI, mock, local, etc.).

    Parameters
    ----------
    adapter:
        Any ``BaseLLMAdapter`` implementation.
    max_retries:
        Additional attempts after the first failure (default 2,
        so 3 total).
    """

    def __init__(
        self,
        adapter: BaseLLMAdapter,
        *,
        max_retries: int = 2,
    ) -> None:
        self._adapter = adapter
        self._max_retries = max_retries

    def analyze(self, description: str) -> BusinessContext:
        """Parse a free-text business description into structured context.

        Parameters
        ----------
        description:
            Plain-English description of the business, e.g.
            ``"AI SaaS for healthcare clinics in India"``.

        Returns
        -------
        BusinessContext
            Validated structured output.

        Raises
        ------
        ContextAnalyzerError
            If the LLM fails to produce valid output after all retries.
        ValueError
            If *description* is empty or whitespace-only.
        """
        description = description.strip()
        if not description:
            raise ValueError("Business description must not be empty.")

        prompt = self._build_prompt(description)
        total_attempts = 1 + self._max_retries
        last_error = ""

        for attempt in range(1, total_attempts + 1):
            raw = self._adapter.generate(prompt)
            try:
                parsed = self._parse_json(raw)
                context = BusinessContext(**parsed)
                if attempt > 1:
                    logger.info(
                        "Context extraction succeeded on attempt %d/%d",
                        attempt,
                        total_attempts,
                    )
                return context

            except (json.JSONDecodeError, TypeError) as exc:
                last_error = f"JSON parse error: {exc}"
                logger.warning(
                    "Attempt %d/%d — %s", attempt, total_attempts, last_error,
                )

            except Exception as exc:  # noqa: BLE001 — Pydantic validation
                last_error = f"Validation error: {exc}"
                logger.warning(
                    "Attempt %d/%d — %s", attempt, total_attempts, last_error,
                )

        raise ContextAnalyzerError(attempts=total_attempts, last_error=last_error)

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _build_prompt(description: str) -> str:
        return (
            _SYSTEM_PROMPT
            + "\n\n"
            + _USER_TEMPLATE.format(description=description)
        )

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Strip markdown fences if the model wraps its response."""
        text = raw.strip()
        if text.startswith("```"):
            # Remove opening fence (```json or ```)
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return json.loads(text.strip())
