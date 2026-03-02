"""Search provider interfaces and contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict


class SearchResultItem(TypedDict):
    """One normalized search result item."""

    title: str
    url: str
    description: str


class SearchResponsePayload(TypedDict):
    """Normalized response contract for search providers."""

    query: str
    results: list[SearchResultItem]


class SearchProvider(ABC):
    """Abstract provider contract for search-only retrieval."""

    @abstractmethod
    def search(self, query: str, *, limit: int = 10) -> SearchResponsePayload:
        """Return normalized search results for the query."""
