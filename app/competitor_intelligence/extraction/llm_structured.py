"""LLM-backed structured extractor with strict schema validation."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any, Protocol

from app.competitor_intelligence.extraction.deterministic import DeterministicExtractor
from app.competitor_intelligence.schemas import (
    ExtractionResult,
    ScrapedDocument,
)


class StructuredLLMClient(Protocol):
    """Abstraction for an LLM that can return raw text responses."""

    async def generate(self, prompt: str) -> str:
        """Generate completion for the supplied prompt."""


class LLMStructuredExtractor:
    """LLM extractor constrained to strict structured JSON outputs."""

    def __init__(
        self,
        *,
        llm_client: StructuredLLMClient,
        fallback: DeterministicExtractor | None = None,
        max_docs: int = 6,
        max_chars_per_doc: int = 3000,
    ) -> None:
        self._llm_client = llm_client
        self._fallback = fallback or DeterministicExtractor()
        self._max_docs = max_docs
        self._max_chars_per_doc = max_chars_per_doc

    async def extract(
        self,
        *,
        competitor_name: str,
        documents: Iterable[ScrapedDocument],
    ) -> ExtractionResult:
        docs = [doc for doc in documents if not doc.error and doc.text][: self._max_docs]
        if not docs:
            return await self._fallback.extract(competitor_name=competitor_name, documents=[])

        prompt = self._build_prompt(competitor_name=competitor_name, documents=docs)
        try:
            raw = await self._llm_client.generate(prompt)
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("LLM output is not an object.")
            payload["competitor_name"] = competitor_name
            payload["extraction_method"] = "llm_structured"
            payload["extracted_at"] = datetime.now(timezone.utc).isoformat()
            return ExtractionResult.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            fallback = await self._fallback.extract(competitor_name=competitor_name, documents=docs)
            return fallback.model_copy(
                update={
                    "warnings": [*fallback.warnings, f"LLM extraction fallback: {exc}"],
                }
            )

    def _build_prompt(self, *, competitor_name: str, documents: list[ScrapedDocument]) -> str:
        document_payload: list[dict[str, Any]] = []
        for doc in documents:
            document_payload.append(
                {
                    "url": str(doc.url),
                    "title": doc.title,
                    "text": doc.text[: self._max_chars_per_doc],
                }
            )
        schema = ExtractionResult.model_json_schema()
        # The extractor output is strictly validated after generation.
        return (
            "You are a data extraction engine.\n"
            "Return strictly valid JSON only.\n"
            "Do not add commentary.\n"
            "Use evidence directly present in the documents.\n"
            "Return all numeric values as numbers.\n\n"
            f"Target competitor: {competitor_name}\n"
            f"Output schema:\n{json.dumps(schema, indent=2)}\n\n"
            f"Documents:\n{json.dumps(document_payload, indent=2)}\n"
        )


class AsyncOpenAIJsonClient:
    """Small async OpenAI wrapper used by the structured extractor."""

    def __init__(self, *, model: str, api_key: str) -> None:
        self._model = model
        self._api_key = api_key

    async def generate(self, prompt: str) -> str:
        from openai import OpenAI  # type: ignore[import-untyped]

        def _run() -> str:
            client = OpenAI(api_key=self._api_key)
            base_kwargs = {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "top_p": 1,
                "response_format": {"type": "json_object"},
            }
            try:
                response = client.chat.completions.create(
                    **base_kwargs,
                    max_completion_tokens=1800,
                )
            except TypeError:
                response = client.chat.completions.create(
                    **base_kwargs,
                    max_tokens=1800,
                )
            except Exception as exc:  # noqa: BLE001
                message = str(exc).lower()
                if "unsupported parameter" in message and "max_completion_tokens" in message:
                    response = client.chat.completions.create(
                        **base_kwargs,
                        max_tokens=1800,
                    )
                else:
                    raise
            return response.choices[0].message.content or "{}"

        return await asyncio.to_thread(_run)
