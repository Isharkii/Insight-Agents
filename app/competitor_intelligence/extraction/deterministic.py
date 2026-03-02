"""Deterministic extraction of competitor signals from scraped text."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from statistics import mean
from typing import Iterable

from app.competitor_intelligence.schemas import (
    ExtractionResult,
    ExtractionSignal,
    ScrapedDocument,
)

_PRICE_PATTERN = re.compile(r"\$(\d{1,5}(?:\.\d{1,2})?)")
_PERCENT_PATTERN = re.compile(r"(\d{1,3}(?:\.\d{1,2})?)\s*%")
_CHURN_HINT = re.compile(r"\bchurn\b", re.IGNORECASE)
_GROWTH_HINT = re.compile(r"\bgrowth\b", re.IGNORECASE)


class DeterministicExtractor:
    """Rule-based extractor with strict reproducible behavior."""

    async def extract(
        self,
        *,
        competitor_name: str,
        documents: Iterable[ScrapedDocument],
    ) -> ExtractionResult:
        now = datetime.now(timezone.utc)
        signals: list[ExtractionSignal] = []
        warnings: list[str] = []
        docs = list(documents)
        if not docs:
            warnings.append("No scraped documents available for deterministic extraction.")

        for doc in docs:
            if doc.error:
                warnings.append(f"Skipped errored document: {doc.url}")
                continue
            if not doc.text:
                continue
            doc_signals = self._signals_from_document(doc)
            signals.extend(doc_signals)

        return ExtractionResult(
            competitor_name=competitor_name,
            extraction_method="deterministic",
            extracted_at=now,
            signals=signals,
            warnings=warnings,
        )

    def _signals_from_document(self, doc: ScrapedDocument) -> list[ExtractionSignal]:
        signals: list[ExtractionSignal] = []
        text = doc.text
        prices = [float(match.group(1)) for match in _PRICE_PATTERN.finditer(text)]
        percents = [float(match.group(1)) for match in _PERCENT_PATTERN.finditer(text)]

        if prices:
            signals.append(
                ExtractionSignal(
                    metric_name="listed_price_usd_mean",
                    value=round(mean(prices), 6),
                    unit="usd",
                    signal_type="competitor_metric",
                    confidence=0.45,
                    source_url=doc.url,
                    evidence=f"Detected {len(prices)} price mention(s).",
                )
            )

        if _CHURN_HINT.search(text) and percents:
            signals.append(
                ExtractionSignal(
                    metric_name="churn_rate_mentioned_pct",
                    value=round(percents[0] / 100.0, 6),
                    unit="ratio",
                    signal_type="peer_metric",
                    confidence=0.35,
                    source_url=doc.url,
                    evidence="Percent near churn-related keyword.",
                )
            )

        if _GROWTH_HINT.search(text) and percents:
            signals.append(
                ExtractionSignal(
                    metric_name="growth_rate_mentioned_pct",
                    value=round(max(percents) / 100.0, 6),
                    unit="ratio",
                    signal_type="industry_metric",
                    confidence=0.3,
                    source_url=doc.url,
                    evidence="Percent near growth-related keyword.",
                )
            )

        return signals
