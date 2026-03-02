"""Tavily search API provider adapter."""

from __future__ import annotations

from datetime import datetime, timezone

from app.competitor_intelligence.config import SearchProviderApiConfig
from app.competitor_intelligence.providers.utils import normalize_domain, request_json
from app.competitor_intelligence.schemas import SearchRequest, SearchResponse, SourceDocument


class TavilySearchProvider:
    """SearchProvider implementation backed by Tavily."""

    def __init__(self, *, config: SearchProviderApiConfig) -> None:
        self._config = config

    @property
    def name(self) -> str:
        return "tavily"

    async def search(self, request: SearchRequest) -> SearchResponse:
        payload: dict[str, object] = {
            "api_key": self._config.api_key,
            "query": request.query,
            "max_results": request.limit,
            "topic": "general",
            "search_depth": "advanced",
            "include_answer": False,
            "include_raw_content": False,
        }
        if request.recency_days is not None:
            payload["days"] = request.recency_days

        body = await request_json(
            method="POST",
            url=self._config.endpoint,
            timeout_seconds=self._config.timeout_seconds,
            headers={"Content-Type": "application/json"},
            json_payload=payload,
        )
        results = body.get("results", [])
        documents: list[SourceDocument] = []
        for idx, item in enumerate(results, start=1):
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            published_at = None
            raw_published = str(item.get("published_date") or "").strip()
            if raw_published:
                try:
                    published_at = datetime.fromisoformat(raw_published.replace("Z", "+00:00"))
                except ValueError:
                    published_at = None
            documents.append(
                SourceDocument(
                    provider=self.name,
                    rank=idx,
                    url=url,
                    title=str(item.get("title") or ""),
                    snippet=str(item.get("content") or ""),
                    published_at=published_at,
                    domain=normalize_domain(url),
                )
            )

        return SearchResponse(
            provider=self.name,
            query=request.query,
            fetched_at=datetime.now(timezone.utc),
            cache_hit=False,
            documents=documents[: request.limit],
        )
