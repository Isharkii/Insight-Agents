"""Synthesize an IntelligenceBundle into a structured InsightBlock.

The LLM receives only the pre-computed signals and business context — it
must never invent numbers.  Every narrative claim must reference a signal
ID (``SIG-<index>``) from the provided data.  The output is validated
against a strict Pydantic schema before being returned.
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.business_intelligence.intelligence_orchestrator import (
    IntelligenceBundle,
    SignalRecord,
)
from llm_synthesis.adapter import BaseLLMAdapter

logger = logging.getLogger(__name__)

_MAX_SECTION_CHARS = int(os.getenv("PROMPT_SECTION_CHAR_LIMIT", "6000"))


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class SignalReference(BaseModel):
    """A single reference tying a narrative claim to its source signal."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    signal_id: str = Field(
        min_length=5,
        max_length=20,
        description="Signal ID in the format SIG-<index> (e.g. 'SIG-0').",
    )
    metric_name: str = Field(min_length=1, max_length=120)
    value: float
    unit: str = Field(default="ratio", min_length=1, max_length=32)


class EmergingSignal(BaseModel):
    """One emerging signal surfaced from the bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(min_length=5, max_length=200)
    description: str = Field(min_length=10, max_length=500)
    supporting_signals: List[SignalReference] = Field(min_length=1, max_length=5)
    relevance: Literal["high", "medium", "low"]


class Zone(BaseModel):
    """An opportunity or risk zone derived from signals."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(min_length=5, max_length=200)
    description: str = Field(min_length=10, max_length=500)
    supporting_signals: List[SignalReference] = Field(min_length=1, max_length=5)


class InsightBlock(BaseModel):
    """Structured synthesis output from an IntelligenceBundle.

    Every narrative field must be traceable to at least one ``SIG-<n>``
    signal ID from the input bundle.  The schema enforces this structurally
    via required ``supporting_signals`` on each item.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    emerging_signals: List[EmergingSignal] = Field(
        min_length=1,
        max_length=10,
        description="Newly surfaced or noteworthy signals.",
    )
    macro_summary: str = Field(
        min_length=10,
        max_length=1000,
        description=(
            "Plain-text summary of macro environment based on news "
            "and trend signals.  Must reference signal IDs."
        ),
    )
    opportunity_zones: List[Zone] = Field(
        min_length=1,
        max_length=10,
        description="Market or strategic opportunities supported by signal evidence.",
    )
    risk_zones: List[Zone] = Field(
        min_length=1,
        max_length=10,
        description="Identified risks supported by signal evidence.",
    )
    momentum_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Composite momentum indicator (0=stalling, 1=strong momentum).",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Synthesis confidence — must not exceed the deterministic "
            "confidence from the input bundle."
        ),
    )

    @model_validator(mode="after")
    def _validate_signal_ids_and_confidence(self) -> "InsightBlock":
        all_refs: list[SignalReference] = []
        for sig in self.emerging_signals:
            all_refs.extend(sig.supporting_signals)
        for zone in self.opportunity_zones:
            all_refs.extend(zone.supporting_signals)
        for zone in self.risk_zones:
            all_refs.extend(zone.supporting_signals)

        # Every signal reference must use the SIG-<n> pattern.
        for ref in all_refs:
            if not ref.signal_id.startswith("SIG-"):
                raise ValueError(
                    f"Signal ID '{ref.signal_id}' does not match required "
                    f"pattern 'SIG-<index>'."
                )

        # macro_summary must mention at least one SIG-<n>.
        if "SIG-" not in self.macro_summary:
            raise ValueError(
                "macro_summary must reference at least one signal ID (SIG-<n>)."
            )

        return self


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a business intelligence analyst that synthesizes structured signal \
data into strategic insight blocks.

ABSOLUTE RULES — violation causes rejection:
1. Return ONLY valid JSON matching the schema below.  No markdown fences.
2. Do NOT invent, estimate, or hallucinate ANY numbers, metrics, or values.
   You may ONLY use the exact values from the PROVIDED SIGNALS table.
3. Every ``supporting_signals`` entry MUST reference a signal ID from the \
   PROVIDED SIGNALS table using the exact format "SIG-<index>".
4. ``macro_summary`` MUST reference at least one "SIG-<n>" signal ID.
5. ``momentum_score`` must be between 0.0 and 1.0 and reflect the overall \
   directional strength evident in the signals — not an invented forecast.
6. ``confidence`` MUST NOT exceed the BUNDLE CONFIDENCE value provided below.

TONE RULES (governed by bundle confidence):
- confidence >= 0.8  → definitive language
- 0.6 <= confidence < 0.8  → cautious language ("data suggests", "appears to")
- 0.4 <= confidence < 0.6  → hedged language ("limited evidence indicates")
- confidence < 0.4  → strongly hedged ("insufficient data", "tentative")

CONTENT RULES:
- ``emerging_signals``: surface the most noteworthy data from the signals.
- ``macro_summary``: describe the external environment using news + trend signals.
- ``opportunity_zones``: strategic opportunities grounded in signal evidence.
- ``risk_zones``: concrete risks grounded in signal evidence.
- Each zone title must be specific, not generic boilerplate.
"""

_SCHEMA_BLOCK = """\
# OUTPUT JSON SCHEMA
```json
{schema}
```
"""

_SIGNALS_BLOCK = """\
# PROVIDED SIGNALS (reference these by ID)
{table}
"""

_CONTEXT_BLOCK = """\
# BUSINESS CONTEXT
```json
{context}
```
"""

_QUALITY_BLOCK = """\
# DATA QUALITY
```json
{quality}
```
"""

_TASK_BLOCK = """\
# TASK
Synthesize the provided signals and business context into a single JSON \
object matching the schema above.  Reference signal IDs.  Do not fabricate \
numbers.
"""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class InsightSynthesizerError(Exception):
    """Raised when synthesis fails after all retry attempts."""

    def __init__(self, attempts: int, last_error: str) -> None:
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"Insight synthesis failed after {attempts} attempt(s): {last_error}"
        )


class InsightSynthesizer:
    """Transform an IntelligenceBundle into a structured InsightBlock.

    Parameters
    ----------
    adapter:
        Any ``BaseLLMAdapter`` implementation.
    max_retries:
        Additional attempts after the first validation failure (default 2).
    """

    def __init__(
        self,
        adapter: BaseLLMAdapter,
        *,
        max_retries: int = 2,
    ) -> None:
        self._adapter = adapter
        self._max_retries = max_retries

    def synthesize(self, bundle: IntelligenceBundle) -> InsightBlock:
        """Synthesize an IntelligenceBundle into a validated InsightBlock.

        Parameters
        ----------
        bundle:
            The output of ``IntelligenceOrchestrator.run()``.

        Returns
        -------
        InsightBlock
            Validated structured output with signal references.

        Raises
        ------
        InsightSynthesizerError
            If the LLM fails to produce valid output after all retries.
        ValueError
            If the bundle has no signals to synthesize.
        """
        if not bundle.signals:
            raise ValueError(
                "Cannot synthesize: IntelligenceBundle contains no signals."
            )

        prompt = self._build_prompt(bundle)
        total_attempts = 1 + self._max_retries
        last_error = ""

        for attempt in range(1, total_attempts + 1):
            raw = self._adapter.generate(prompt)
            try:
                parsed = _parse_json(raw)
                block = InsightBlock(**parsed)
                _validate_signal_ids_exist(block, bundle.signals)
                _validate_confidence_ceiling(block, bundle.confidence)
                if attempt > 1:
                    logger.info(
                        "Insight synthesis succeeded on attempt %d/%d",
                        attempt,
                        total_attempts,
                    )
                return block

            except (json.JSONDecodeError, TypeError) as exc:
                last_error = f"JSON parse error: {exc}"
                logger.warning(
                    "Synthesis attempt %d/%d — %s",
                    attempt, total_attempts, last_error,
                )

            except Exception as exc:  # noqa: BLE001
                last_error = f"Validation error: {exc}"
                logger.warning(
                    "Synthesis attempt %d/%d — %s",
                    attempt, total_attempts, last_error,
                )

        raise InsightSynthesizerError(
            attempts=total_attempts, last_error=last_error,
        )

    # -- prompt construction ------------------------------------------------

    def _build_prompt(self, bundle: IntelligenceBundle) -> str:
        parts: list[str] = [_SYSTEM_PROMPT, ""]

        # Business context
        if bundle.business_context is not None:
            ctx_json = bundle.business_context.model_dump_json(indent=2)
            parts.append(
                _CONTEXT_BLOCK.format(context=_truncate(ctx_json))
            )

        # Signals table (each signal gets a stable ID)
        parts.append(
            _SIGNALS_BLOCK.format(table=_format_signal_table(bundle.signals))
        )

        # Data quality
        quality = {
            "bundle_confidence": round(bundle.confidence, 4),
            "total_signals": len(bundle.signals),
            "stage_summary": {
                s.stage: s.status for s in bundle.stage_statuses
            },
            "warnings": bundle.warnings[:20],
        }
        if bundle.confidence >= 0.8:
            quality["tone_directive"] = "definitive"
        elif bundle.confidence >= 0.6:
            quality["tone_directive"] = "cautious"
        elif bundle.confidence >= 0.4:
            quality["tone_directive"] = "hedged"
        else:
            quality["tone_directive"] = "strongly_hedged"
        parts.append(
            _QUALITY_BLOCK.format(
                quality=_truncate(json.dumps(quality, indent=2))
            )
        )

        # Schema
        parts.append(
            _SCHEMA_BLOCK.format(
                schema=json.dumps(InsightBlock.model_json_schema(), indent=2)
            )
        )

        # Task
        parts.append(_TASK_BLOCK)

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _parse_json(raw: str) -> dict:
    """Strip markdown fences if present, then parse JSON."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def _validate_signal_ids_exist(
    block: InsightBlock,
    signals: List[SignalRecord],
) -> None:
    """Ensure every referenced SIG-<n> actually exists in the bundle."""
    valid_ids = {f"SIG-{i}" for i in range(len(signals))}

    all_refs: list[SignalReference] = []
    for sig in block.emerging_signals:
        all_refs.extend(sig.supporting_signals)
    for zone in block.opportunity_zones:
        all_refs.extend(zone.supporting_signals)
    for zone in block.risk_zones:
        all_refs.extend(zone.supporting_signals)

    bad_ids = {ref.signal_id for ref in all_refs if ref.signal_id not in valid_ids}
    if bad_ids:
        raise ValueError(
            f"Signal IDs not found in bundle: {sorted(bad_ids)}. "
            f"Valid IDs: SIG-0 through SIG-{len(signals) - 1}."
        )


def _validate_confidence_ceiling(
    block: InsightBlock,
    bundle_confidence: float,
) -> None:
    """InsightBlock confidence must not exceed the deterministic bundle confidence."""
    if block.confidence > bundle_confidence + 1e-6:
        raise ValueError(
            f"InsightBlock confidence ({block.confidence:.4f}) exceeds "
            f"bundle confidence ceiling ({bundle_confidence:.4f})."
        )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_signal_table(signals: List[SignalRecord]) -> str:
    """Format signals as a JSON array with stable IDs for LLM consumption."""
    rows = []
    for i, sig in enumerate(signals):
        rows.append({
            "id": f"SIG-{i}",
            "source": sig.source,
            "metric_name": sig.metric_name,
            "value": sig.value,
            "unit": sig.unit,
            "confidence": sig.confidence,
            "evidence": sig.evidence,
        })
    body = json.dumps(rows, indent=2)
    return _truncate(body)


def _truncate(text: str) -> str:
    if len(text) <= _MAX_SECTION_CHARS:
        return text
    return text[:_MAX_SECTION_CHARS] + "\n... (truncated)"
