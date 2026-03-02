"""HTTP scraper implementation isolated from search providers."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from app.competitor_intelligence.config import ScraperConfig
from app.competitor_intelligence.schemas import ScrapedDocument


class RequestsScraper:
    """Scraper implementation using requests + readability cleanup."""

    def __init__(self, *, config: ScraperConfig) -> None:
        self._config = config

    async def fetch(self, url: str) -> ScrapedDocument:
        def _run() -> ScrapedDocument:
            headers = {"User-Agent": self._config.user_agent}
            fetched_at = datetime.now(timezone.utc)
            if self._is_domain_blacklisted(url):
                return ScrapedDocument(
                    url=url,
                    fetched_at=fetched_at,
                    status_code=0,
                    title="",
                    content_type="",
                    text="",
                    metadata={"blocked": True},
                    error="domain_blacklisted",
                )
            try:
                response = requests.get(
                    url,
                    headers=headers,
                    timeout=self._config.timeout_seconds,
                )
                status_code = int(response.status_code)
                content_type = str(response.headers.get("Content-Type") or "")
                html = response.text if response.text else ""
                title, text, metadata = self._normalize_html(html)
                return ScrapedDocument(
                    url=url,
                    fetched_at=fetched_at,
                    status_code=status_code,
                    title=title,
                    content_type=content_type,
                    text=text,
                    metadata=metadata,
                    error=None,
                )
            except requests.RequestException as exc:
                return ScrapedDocument(
                    url=url,
                    fetched_at=fetched_at,
                    status_code=0,
                    title="",
                    content_type="",
                    text="",
                    metadata={},
                    error=str(exc),
                )

        return await asyncio.to_thread(_run)

    async def fetch_many(self, urls: Sequence[str], *, max_concurrency: int) -> list[ScrapedDocument]:
        semaphore = asyncio.Semaphore(max(1, max_concurrency))

        async def _bounded_fetch(url: str) -> ScrapedDocument:
            async with semaphore:
                return await self.fetch(url)

        tasks = [_bounded_fetch(url) for url in urls]
        return await asyncio.gather(*tasks)

    def _is_domain_blacklisted(self, url: str) -> bool:
        host = (urlparse(url).hostname or "").strip().lower()
        if not host:
            return False
        blacklist = [item.strip().lower() for item in self._config.domain_blacklist if item.strip()]
        for blocked in blacklist:
            if host == blocked or host.endswith(f".{blocked}"):
                return True
        return False

    def _normalize_html(self, html: str) -> tuple[str, str, dict[str, Any]]:
        if not html:
            return "", "", {}
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()[:1000]
        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        text = text[: self._config.max_text_chars]
        metadata = {
            "text_chars": len(text),
            "truncated": len(text) >= self._config.max_text_chars,
        }
        return title, text, metadata
