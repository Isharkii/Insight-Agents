"""Strict LLM extractor for competitor intelligence profiles."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)


class CompetitorIntelligence(BaseModel):
    """Validated competitor profile extracted from scraped text."""

    model_config = ConfigDict(extra="forbid")

    company_name: str = Field(min_length=1, max_length=255)
    pricing_model: str | None = None
    target_segment: str | None = None
    key_features: list[str] = Field(default_factory=list)
    positioning: str | None = None
    funding_status: str | None = None
    estimated_scale: str | None = None
    confidence_score: float = Field(ge=0.0, le=1.0)


class CompetitorIntelligenceResponse(BaseModel):
    """Structured non-throwing response wrapper for extraction."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(pattern="^(success|partial|failed)$")
    error: str | None = None
    data: CompetitorIntelligence | None = None


class LLMJsonClient(Protocol):
    """Protocol for JSON-only LLM generation clients."""

    async def generate_json(self, prompt: str) -> str:
        """Return a JSON object string for the provided prompt."""


class OpenAIJsonClient:
    """OpenAI-backed JSON-only client."""

    def __init__(self, *, model: str, api_key: str) -> None:
        self._model = model
        self._api_key = api_key

    async def generate_json(self, prompt: str) -> str:
        from openai import OpenAI  # type: ignore[import-untyped]

        def _run() -> str:
            client = OpenAI(api_key=self._api_key)
            response = client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                top_p=1,
                max_tokens=900,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content or "{}"

        try:
            return await asyncio.to_thread(_run)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"LLM request failed: {exc}") from exc


class CompetitorIntelligenceExtractor:
    """Extracts strict structured competitor profile from raw scraped text."""

    def __init__(
        self,
        *,
        llm_client: LLMJsonClient,
        max_retries: int = 2,
        max_text_length: int = 12000,
        failure_threshold: int = 5,
        circuit_reset_seconds: int = 120,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0.")
        if max_text_length <= 0:
            raise ValueError("max_text_length must be > 0.")
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be > 0.")
        if circuit_reset_seconds <= 0:
            raise ValueError("circuit_reset_seconds must be > 0.")
        self._llm_client = llm_client
        self._max_retries = max_retries
        self._max_text_length = max_text_length
        self._failure_threshold = failure_threshold
        self._circuit_reset_seconds = circuit_reset_seconds
        self._failure_count = 0
        self._circuit_open_until = 0.0
        self._state_lock = asyncio.Lock()

    async def extract(self, *, raw_text: str, company_name: str) -> CompetitorIntelligence:
        response = await self.extract_with_status(raw_text=raw_text, company_name=company_name)
        if response.data is not None:
            return response.data
        # Hard fallback: never propagate exceptions to callers.
        return self._fallback_from_text(raw_text=raw_text, company_name=company_name)

    async def extract_with_status(
        self,
        *,
        raw_text: str,
        company_name: str,
    ) -> CompetitorIntelligenceResponse:
        """Non-throwing extraction entrypoint with structured status/error/data."""
        try:
            normalized_company = str(company_name or "").strip()
            normalized_text = str(raw_text or "").strip()
            if not normalized_text:
                return CompetitorIntelligenceResponse(
                    status="failed",
                    error="raw_text must not be empty.",
                    data=None,
                )
            if not normalized_company:
                return CompetitorIntelligenceResponse(
                    status="failed",
                    error="company_name must not be empty.",
                    data=None,
                )

            truncated = normalized_text[: self._max_text_length]
            if await self._is_circuit_open():
                fallback = self._fallback_from_text(
                    raw_text=truncated,
                    company_name=normalized_company,
                )
                return CompetitorIntelligenceResponse(
                    status="partial",
                    error="circuit_breaker_open: skipping LLM extraction temporarily.",
                    data=fallback,
                )

            prompt = self._build_prompt(raw_text=truncated, company_name=normalized_company)
            attempts = self._max_retries + 1
            last_error: Exception | None = None

            for _ in range(attempts):
                try:
                    raw_response = await self._llm_client.generate_json(prompt)
                    parsed = json.loads(raw_response)
                    if not isinstance(parsed, dict):
                        raise ValueError("LLM response was not a JSON object.")
                    model = CompetitorIntelligence.model_validate(parsed)
                    await self._record_success()
                    return CompetitorIntelligenceResponse(status="success", error=None, data=model)
                except (json.JSONDecodeError, ValidationError, ValueError, RuntimeError) as exc:
                    last_error = exc
                    await self._record_failure()
                    logger.warning("competitor_extraction_attempt_failed company=%s error=%s", normalized_company, exc)

            fallback = self._fallback_from_text(raw_text=truncated, company_name=normalized_company)
            return CompetitorIntelligenceResponse(
                status="partial",
                error=f"llm_extraction_failed:{last_error}",
                data=fallback,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("competitor_extraction_unexpected_error company=%s error=%s", company_name, exc)
            return CompetitorIntelligenceResponse(
                status="failed",
                error=f"unexpected_error:{exc}",
                data=None,
            )

    def _build_prompt(self, *, raw_text: str, company_name: str) -> str:
        schema_json = json.dumps(CompetitorIntelligence.model_json_schema(), indent=2)
        return (
            "You are a strict data extraction engine.\n"
            "Output only one valid JSON object.\n"
            "Do not output markdown.\n"
            "Do not output explanations.\n"
            "If a field is unknown, use null or [] as appropriate.\n"
            "confidence_score must be between 0 and 1.\n\n"
            f"Target company: {company_name}\n"
            f"Schema:\n{schema_json}\n\n"
            f"Raw scraped text:\n{raw_text}\n"
        )

    async def _is_circuit_open(self) -> bool:
        async with self._state_lock:
            return time.monotonic() < self._circuit_open_until

    async def _record_success(self) -> None:
        async with self._state_lock:
            self._failure_count = 0
            self._circuit_open_until = 0.0

    async def _record_failure(self) -> None:
        async with self._state_lock:
            self._failure_count += 1
            if self._failure_count >= self._failure_threshold:
                self._circuit_open_until = time.monotonic() + float(self._circuit_reset_seconds)
                self._failure_count = 0

    def _fallback_from_text(self, *, raw_text: str, company_name: str) -> CompetitorIntelligence:
        lowered = raw_text.lower()
        pricing_model = None
        if "freemium" in lowered:
            pricing_model = "freemium"
        elif "subscription" in lowered:
            pricing_model = "subscription"
        elif "enterprise" in lowered and "pricing" in lowered:
            pricing_model = "enterprise"

        target_segment = None
        if "enterprise" in lowered:
            target_segment = "enterprise"
        elif "smb" in lowered or "small business" in lowered:
            target_segment = "smb"

        key_features: list[str] = []
        for token in ("analytics", "automation", "api", "integrations", "security", "workflow"):
            if token in lowered:
                key_features.append(token)

        positioning = "cost_leader" if "affordable" in lowered or "low cost" in lowered else None
        funding_status = "unknown"
        estimated_scale = "unknown"

        return CompetitorIntelligence(
            company_name=company_name.strip() or "unknown",
            pricing_model=pricing_model,
            target_segment=target_segment,
            key_features=key_features,
            positioning=positioning,
            funding_status=funding_status,
            estimated_scale=estimated_scale,
            confidence_score=0.25,
        )
