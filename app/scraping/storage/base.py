"""
Storage layer interfaces for canonical scraped insights.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.domain.canonical_insight import CanonicalInsightInput


class InsightStorage(ABC):
    """
    Storage abstraction for canonical insight writes.
    """

    @abstractmethod
    def store(self, rows: Sequence[CanonicalInsightInput]) -> int:
        """
        Persist canonical rows and return inserted row count.
        """
