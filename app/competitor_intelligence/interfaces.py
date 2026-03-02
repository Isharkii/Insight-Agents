"""Core abstraction contracts for modular competitor intelligence."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from app.competitor_intelligence.schemas import (
    ExtractionResult,
    ScrapedDocument,
    SearchRequest,
    SearchResponse,
)


@runtime_checkable
class SearchProvider(Protocol):
    """Provider abstraction for web search APIs."""

    @property
    def name(self) -> str:
        """Unique provider name (for example brave, serper, tavily)."""

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Search for documents matching query constraints."""


@runtime_checkable
class Scraper(Protocol):
    """Scraper abstraction kept separate from search providers."""

    async def fetch(self, url: str) -> ScrapedDocument:
        """Fetch and normalize one URL."""

    async def fetch_many(self, urls: Sequence[str], *, max_concurrency: int) -> list[ScrapedDocument]:
        """Fetch and normalize several URLs with bounded concurrency."""


@runtime_checkable
class Extractor(Protocol):
    """Structured extraction abstraction (deterministic or LLM-backed)."""

    async def extract(
        self,
        *,
        competitor_name: str,
        documents: Sequence[ScrapedDocument],
    ) -> ExtractionResult:
        """Extract competitor signals into strict structured JSON."""
