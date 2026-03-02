"""
Async-compatible URL text scraper with trafilatura extraction.

This module is intentionally independent from search providers and LLM layers.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS: Final[float] = 10.0
_DEFAULT_USER_AGENT: Final[str] = "InsightAgentTextScraper/1.0 (+https://example.com/bot)"
_BLOCKED_HOSTS: Final[set[str]] = {"localhost", "127.0.0.1", "::1"}
_DISALLOWED_STATUS_CODES: Final[set[int]] = {403, 404, 500}


@dataclass(slots=True)
class AsyncTextScraper:
    """Fetches URLs safely and extracts cleaned plain text via trafilatura."""

    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
    user_agent: str = _DEFAULT_USER_AGENT

    async def fetch_text(self, url: str) -> str | None:
        """
        Fetch a URL and return cleaned plain text.

        Returns None when validation, fetch, or extraction fails.
        """
        if not _is_valid_url(url):
            logger.warning("Text scraper rejected invalid url: %s", url)
            return None

        raw_html, status_code = await asyncio.to_thread(self._download_html, url)
        if raw_html is None:
            if status_code in _DISALLOWED_STATUS_CODES:
                logger.info("Text scraper skipped url=%s status=%s", url, status_code)
            return None

        return await asyncio.to_thread(_extract_clean_text, raw_html)

    async def fetch_many(self, urls: list[str], *, max_concurrency: int = 5) -> dict[str, str | None]:
        """Fetch and extract multiple URLs with bounded concurrency."""
        semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
        results: dict[str, str | None] = {}

        async def _runner(url: str) -> None:
            async with semaphore:
                results[url] = await self.fetch_text(url)

        await asyncio.gather(*(_runner(url) for url in urls))
        return results

    def _download_html(self, url: str) -> tuple[str | None, int | None]:
        headers = {"User-Agent": self.user_agent}
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=self.timeout_seconds,
                allow_redirects=True,
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            logger.warning("Text scraper network error url=%s error=%s", url, exc)
            return None, None
        except requests.RequestException as exc:
            logger.warning("Text scraper request error url=%s error=%s", url, exc)
            return None, None

        status_code = int(response.status_code)
        if status_code in _DISALLOWED_STATUS_CODES:
            return None, status_code
        if status_code >= 400:
            logger.warning("Text scraper http error url=%s status=%s", url, status_code)
            return None, status_code

        content_type = str(response.headers.get("Content-Type") or "").lower()
        if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
            logger.info("Text scraper unsupported content type url=%s type=%s", url, content_type)
            return None, status_code

        return response.text, status_code


def _extract_clean_text(raw_html: str) -> str | None:
    """Strip noisy elements and extract main text using trafilatura."""
    try:
        import trafilatura  # type: ignore[import-not-found]
    except ImportError:
        logger.error("trafilatura is not installed; cannot extract article text.")
        return None

    try:
        soup = BeautifulSoup(raw_html, "html.parser")
        # Remove common non-content containers before extraction.
        for tag in soup.select(
            "script, style, noscript, nav, footer, header, aside, form, iframe, [role='navigation'], "
            ".menu, .navbar, .breadcrumbs, .ad, .ads, .advertisement, .cookie, .popup"
        ):
            tag.decompose()
        cleaned_html = str(soup)

        text = trafilatura.extract(
            cleaned_html,
            include_comments=False,
            include_tables=False,
            include_formatting=False,
            favor_precision=True,
            output_format="txt",
        )
        if not text:
            return None
        normalized = re.sub(r"\s+", " ", text).strip()
        return normalized or None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Text extraction failed: %s", exc)
        return None


def _is_valid_url(url: str) -> bool:
    """Basic domain and scheme validation."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    if parsed.scheme not in {"http", "https"}:
        return False
    if not parsed.hostname:
        return False
    host = parsed.hostname.strip().lower()
    if host in _BLOCKED_HOSTS:
        return False
    if "@" in url:
        return False

    try:
        ip_obj = ipaddress.ip_address(host)
    except ValueError:
        # Basic domain sanity: allow FQDN-like names.
        if "." not in host:
            return False
        return True

    if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_multicast or ip_obj.is_reserved or ip_obj.is_link_local:
        return False
    return True
