"""
BeautifulSoup-based parsing layer for competitor pages.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from bs4 import BeautifulSoup, Tag

from app.scraping.types import ParsedCompetitorData

PRICE_REGEX = re.compile(
    r"(?:USD|US\$|\$|EUR|€|GBP|£)?\s?\d{1,5}(?:[.,]\d{1,2})?(?:\s?/\s?(?:mo|month|yr|year|user))?",
    flags=re.IGNORECASE,
)
DATE_PATTERNS = [
    "%Y-%m-%d",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%m/%d/%Y",
    "%d/%m/%Y",
]


class HTMLParsingLayer:
    """
    Deterministic parser utilities for HTML documents.
    """

    @classmethod
    def parse_page(
        cls,
        *,
        page_kind: str,
        page_url: str,
        soup: BeautifulSoup,
        selectors: list[str],
    ) -> ParsedCompetitorData:
        normalized_kind = page_kind.strip().lower()
        if normalized_kind == "pricing":
            return ParsedCompetitorData(
                pricing_items=cls.extract_pricing(
                    soup=soup,
                    selectors=selectors,
                    page_url=page_url,
                )
            )
        if normalized_kind == "products":
            return ParsedCompetitorData(
                product_items=cls.extract_products(
                    soup=soup,
                    selectors=selectors,
                    page_url=page_url,
                )
            )
        if normalized_kind in {"marketing", "headlines"}:
            return ParsedCompetitorData(
                marketing_headlines=cls.extract_headlines(
                    soup=soup,
                    selectors=selectors,
                    page_url=page_url,
                )
            )
        if normalized_kind in {"events", "announcements"}:
            return ParsedCompetitorData(
                event_announcements=cls.extract_events(
                    soup=soup,
                    selectors=selectors,
                    page_url=page_url,
                )
            )
        return ParsedCompetitorData()

    @classmethod
    def extract_pricing(
        cls,
        *,
        soup: BeautifulSoup,
        selectors: list[str],
        page_url: str,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for node in cls._select_elements(soup=soup, selectors=selectors):
            raw_text = cls._clean_text(node.get_text(" ", strip=True))
            if not raw_text:
                continue
            price_match = PRICE_REGEX.search(raw_text)
            if price_match is None:
                continue
            items.append(
                {
                    "plan": cls._extract_title(node),
                    "price_text": price_match.group(0),
                    "details": raw_text,
                    "page_url": page_url,
                }
            )

        if not items:
            for node in soup.find_all(string=PRICE_REGEX):
                text = cls._clean_text(str(node))
                if not text:
                    continue
                match = PRICE_REGEX.search(text)
                if match is None:
                    continue
                items.append(
                    {
                        "plan": "unlabeled",
                        "price_text": match.group(0),
                        "details": text,
                        "page_url": page_url,
                    }
                )
        return cls._dedupe_dicts(items)

    @classmethod
    def extract_products(
        cls,
        *,
        soup: BeautifulSoup,
        selectors: list[str],
        page_url: str,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for node in cls._select_elements(soup=soup, selectors=selectors):
            text = cls._clean_text(node.get_text(" ", strip=True))
            if not text:
                continue
            items.append(
                {
                    "name": cls._extract_title(node),
                    "description": text,
                    "page_url": page_url,
                }
            )
        return cls._dedupe_dicts(items)

    @classmethod
    def extract_headlines(
        cls,
        *,
        soup: BeautifulSoup,
        selectors: list[str],
        page_url: str,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for node in cls._select_elements(soup=soup, selectors=selectors):
            headline = cls._clean_text(node.get_text(" ", strip=True))
            if len(headline) < 8:
                continue
            items.append({"headline": headline, "page_url": page_url})
        return cls._dedupe_dicts(items)

    @classmethod
    def extract_events(
        cls,
        *,
        soup: BeautifulSoup,
        selectors: list[str],
        page_url: str,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for node in cls._select_elements(soup=soup, selectors=selectors):
            text = cls._clean_text(node.get_text(" ", strip=True))
            if len(text) < 8:
                continue
            items.append(
                {
                    "title": cls._extract_title(node),
                    "details": text,
                    "event_date": cls._extract_date_from_text(text),
                    "page_url": page_url,
                }
            )
        return cls._dedupe_dicts(items)

    @classmethod
    def _select_elements(
        cls,
        *,
        soup: BeautifulSoup,
        selectors: list[str],
    ) -> list[Tag]:
        found: list[Tag] = []
        if selectors:
            for selector in selectors:
                found.extend(soup.select(selector))
        else:
            found.extend(soup.find_all(["h1", "h2", "h3", "article", "section", "li"]))
        return found[:300]

    @staticmethod
    def _extract_title(node: Tag) -> str:
        heading = node.find(["h1", "h2", "h3", "h4", "strong", "b"])
        if heading is not None:
            text = heading.get_text(" ", strip=True)
            if text:
                return text[:180]
        node_text = node.get_text(" ", strip=True)
        if not node_text:
            return "untitled"
        return node_text[:180]

    @staticmethod
    def _clean_text(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    @classmethod
    def _extract_date_from_text(cls, value: str) -> str | None:
        compact = cls._clean_text(value)

        for pattern in DATE_PATTERNS:
            try:
                parsed = datetime.strptime(compact, pattern)
                return parsed.replace(tzinfo=timezone.utc).isoformat()
            except ValueError:
                continue

        tokens = re.findall(
            r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|"
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4})\b",
            compact,
            flags=re.IGNORECASE,
        )
        for token in tokens:
            for pattern in DATE_PATTERNS:
                try:
                    parsed = datetime.strptime(token, pattern)
                    return parsed.replace(tzinfo=timezone.utc).isoformat()
                except ValueError:
                    continue
        return None

    @staticmethod
    def _dedupe_dicts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for item in items:
            key = repr(sorted(item.items()))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped[:500]
