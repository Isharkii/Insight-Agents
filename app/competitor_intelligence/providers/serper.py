"""Serper search API provider adapter."""

from __future__ import annotations

from datetime import datetime, timezone

from app.competitor_intelligence.config import SearchProviderApiConfig
from app.competitor_intelligence.providers.utils import normalize_domain, request_json
from app.competitor_intelligence.schemas import SearchRequest, SearchResponse, SourceDocument


class SerperSearchProvider:
    """SearchProvider implementation backed by Serper."""

    def __init__(self, *, config: SearchProviderApiConfig) -> None:
        self._config = config

    @property
    def name(self) -> str:
        return "serper"

    async def search(self, request: SearchRequest) -> SearchResponse:
        headers = {
            "X-API-KEY": self._config.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "q": request.query,
            "num": request.limit,
            "gl": request.market.lower(),
            "hl": request.language,
        }
        if request.recency_days is not None:
            payload["tbs"] = f"qdr:d{request.recency_days}"

        body = await request_json(
            method="POST",
            url=self._config.endpoint,
            timeout_seconds=self._config.timeout_seconds,
            headers=headers,
            json_payload=payload,
        )
        organic = body.get("organic", [])
        documents: list[SourceDocument] = []
        for idx, item in enumerate(organic, start=1):
            if not isinstance(item, dict):
                continue
            url = str(item.get("link") or "").strip()
            if not url:
                continue
            date_text = str(item.get("date") or "").strip()
            published_at = None
            if date_text:
                try:
                    published_at = datetime.fromisoformat(date_text.replace("Z", "+00:00"))
                except ValueError:
                    published_at = None
            documents.append(
                SourceDocument(
                    provider=self.name,
                    rank=idx,
                    url=url,
                    title=str(item.get("title") or ""),
                    snippet=str(item.get("snippet") or ""),
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
