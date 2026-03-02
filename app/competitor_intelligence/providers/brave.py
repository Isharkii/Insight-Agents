"""Brave Search API provider adapter."""

from __future__ import annotations

from datetime import datetime, timezone

from app.competitor_intelligence.config import SearchProviderApiConfig
from app.competitor_intelligence.schemas import SearchRequest, SearchResponse, SourceDocument
from app.competitor_intelligence.providers.utils import normalize_domain, request_json


class BraveSearchProvider:
    """SearchProvider implementation backed by Brave web search."""

    def __init__(self, *, config: SearchProviderApiConfig) -> None:
        self._config = config

    @property
    def name(self) -> str:
        return "brave"

    async def search(self, request: SearchRequest) -> SearchResponse:
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self._config.api_key,
        }
        params: dict[str, object] = {
            "q": request.query,
            "count": request.limit,
            "country": request.market,
            "search_lang": request.language,
        }
        if request.recency_days is not None:
            params["freshness"] = f"pd{request.recency_days}"

        payload = await request_json(
            method="GET",
            url=self._config.endpoint,
            timeout_seconds=self._config.timeout_seconds,
            headers=headers,
            params=params,
        )
        web = payload.get("web")
        results = web.get("results", []) if isinstance(web, dict) else []
        documents: list[SourceDocument] = []
        for idx, item in enumerate(results, start=1):
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            age = str(item.get("age") or "").strip()
            published_at = None
            if age:
                try:
                    published_at = datetime.fromisoformat(age.replace("Z", "+00:00"))
                except ValueError:
                    published_at = None
            documents.append(
                SourceDocument(
                    provider=self.name,
                    rank=idx,
                    url=url,
                    title=str(item.get("title") or ""),
                    snippet=str(item.get("description") or ""),
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
