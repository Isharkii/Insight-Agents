"""Generate a signal-grounded StrategyBlock from an InsightBlock.

Every strategy action must reference at least one signal ID from the
InsightBlock it was derived from.  Generic advice is structurally
rejected by schema validators.
"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.business_intelligence.insight_synthesizer import (
    InsightBlock,
    SignalReference,
)
from llm_synthesis.adapter import BaseLLMAdapter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Blocked patterns — reject boilerplate at the schema level
# ---------------------------------------------------------------------------

_GENERIC_PATTERNS: tuple[str, ...] = (
    "improve performance",
    "focus on growth",
    "monitor closely",
    "optimize operations",
    "leverage synergies",
    "align stakeholders",
    "take action",
    "increase efficiency",
    "drive innovation",
    "enhance capabilities",
    "consider investing",
    "explore opportunities",
)


def _is_generic(text: str) -> bool:
    lowered = re.sub(r"\s+", " ", text.strip().lower())
    return any(pat in lowered for pat in _GENERIC_PATTERNS)


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class StrategyAction(BaseModel):
    """One concrete strategy action grounded in signal evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    action: str = Field(
        min_length=15,
        max_length=500,
        description="Specific, actionable strategy statement.",
    )
    rationale: str = Field(
        min_length=15,
        max_length=500,
        description="Why this action matters — must reference signal evidence.",
    )
    supporting_signals: List[SignalReference] = Field(
        min_length=1,
        max_length=5,
        description="Signal IDs that justify this action.",
    )
    priority: Literal["critical", "high", "medium"] = Field(
        description="Execution priority.",
    )

    @model_validator(mode="after")
    def _reject_generic(self) -> "StrategyAction":
        if _is_generic(self.action):
            raise ValueError(
                f"Generic strategy rejected: '{self.action[:80]}…'. "
                "Actions must be specific and signal-grounded."
            )
        return self


class CompetitiveAngle(BaseModel):
    """Competitive positioning recommendation grounded in signals."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    positioning: str = Field(
        min_length=15,
        max_length=500,
        description="How to position against competitors based on signal evidence.",
    )
    differentiation: str = Field(
        min_length=15,
        max_length=500,
        description="Key differentiation lever identified from signals.",
    )
    supporting_signals: List[SignalReference] = Field(
        min_length=1,
        max_length=5,
    )


class RiskMitigation(BaseModel):
    """Risk mitigation plan grounded in risk zone signals."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    risk_title: str = Field(
        min_length=5,
        max_length=200,
        description="The risk being mitigated (should match a risk_zone title).",
    )
    mitigation: str = Field(
        min_length=15,
        max_length=500,
        description="Concrete mitigation action.",
    )
    supporting_signals: List[SignalReference] = Field(
        min_length=1,
        max_length=5,
    )


class StrategyBlock(BaseModel):
    """Structured strategy output derived from an InsightBlock.

    Every field is traceable to signal IDs.  Generic advice is rejected
    at the schema level.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    short_term_actions: List[StrategyAction] = Field(
        min_length=3,
        max_length=3,
        description="Exactly 3 immediate actions (0–3 month horizon).",
    )
    mid_term_actions: List[StrategyAction] = Field(
        min_length=3,
        max_length=3,
        description="Exactly 3 medium-term actions (3–12 month horizon).",
    )
    long_term_positioning: str = Field(
        min_length=20,
        max_length=800,
        description=(
            "Long-term strategic direction (12+ months). "
            "Must reference signal IDs."
        ),
    )
    competitive_angle: CompetitiveAngle = Field(
        description="Competitive positioning derived from opportunity and risk signals.",
    )
    risk_mitigation: List[RiskMitigation] = Field(
        min_length=1,
        max_length=5,
        description="Mitigation plans for each identified risk zone.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Must not exceed the InsightBlock confidence.",
    )

    @model_validator(mode="after")
    def _validate_signal_refs_and_uniqueness(self) -> "StrategyBlock":
        # Every signal reference must use SIG-<n> format
        all_refs: list[SignalReference] = []
        for act in self.short_term_actions:
            all_refs.extend(act.supporting_signals)
        for act in self.mid_term_actions:
            all_refs.extend(act.supporting_signals)
        all_refs.extend(self.competitive_angle.supporting_signals)
        for rm in self.risk_mitigation:
            all_refs.extend(rm.supporting_signals)

        for ref in all_refs:
            if not ref.signal_id.startswith("SIG-"):
                raise ValueError(
                    f"Signal ID '{ref.signal_id}' does not match 'SIG-<index>'."
                )

        # long_term_positioning must reference at least one signal
        if "SIG-" not in self.long_term_positioning:
            raise ValueError(
                "long_term_positioning must reference at least one signal ID (SIG-<n>)."
            )

        # No duplicate action text across short and mid term
        seen: set[str] = set()
        for act in [*self.short_term_actions, *self.mid_term_actions]:
            normalized = re.sub(r"\s+", " ", act.action.strip().lower())
            if normalized in seen:
                raise ValueError(
                    "Duplicate action detected across strategy horizons."
                )
            seen.add(normalized)

        return self


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a business strategist that converts structured insight data into \
concrete, actionable strategy blocks.

ABSOLUTE RULES — violation causes rejection:
1. Return ONLY valid JSON matching the schema below.  No markdown fences.
2. Do NOT invent, estimate, or hallucinate ANY numbers or metrics.
   Use ONLY the exact values from the PROVIDED DATA.
3. Every ``supporting_signals`` entry MUST reference a signal ID from the \
   PROVIDED DATA using the exact format "SIG-<index>".
4. ``long_term_positioning`` MUST reference at least one "SIG-<n>" ID.
5. ``confidence`` MUST NOT exceed the INSIGHT CONFIDENCE value provided.
6. ``short_term_actions`` must contain EXACTLY 3 actions (0–3 month horizon).
7. ``mid_term_actions`` must contain EXACTLY 3 actions (3–12 month horizon).
8. No action may be duplicated across short-term and mid-term.
9. Every action must be specific and grounded in signal evidence — \
   generic advice like "improve performance", "focus on growth", \
   "monitor closely", "optimize operations" will be rejected.

TONE RULES (governed by insight confidence):
- confidence >= 0.8  → definitive, assertive recommendations
- 0.6 <= confidence < 0.8  → cautious ("data suggests", "consider")
- confidence < 0.6  → hedged ("if validated", "pending further signals")

CONTENT RULES:
- ``short_term_actions``: immediate, executable actions with clear owners.
- ``mid_term_actions``: quarterly initiatives requiring planning.
- ``long_term_positioning``: 12+ month strategic direction statement.
- ``competitive_angle``: how to position versus competitors based on signals.
- ``risk_mitigation``: one plan per risk zone from the insight block.
- Each risk_mitigation.risk_title should correspond to a risk_zone title.
"""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class StrategyGeneratorError(Exception):
    """Raised when strategy generation fails after all retry attempts."""

    def __init__(self, attempts: int, last_error: str) -> None:
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"Strategy generation failed after {attempts} attempt(s): {last_error}"
        )


class StrategyGenerator:
    """Transform an InsightBlock into a validated StrategyBlock.

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

    def generate(self, insight: InsightBlock) -> StrategyBlock:
        """Generate a StrategyBlock from a validated InsightBlock.

        Parameters
        ----------
        insight:
            The output of ``InsightSynthesizer.synthesize()``.

        Returns
        -------
        StrategyBlock
            Validated structured strategy with signal references.

        Raises
        ------
        StrategyGeneratorError
            If the LLM fails to produce valid output after all retries.
        """
        prompt = self._build_prompt(insight)
        total_attempts = 1 + self._max_retries
        last_error = ""

        for attempt in range(1, total_attempts + 1):
            raw = self._adapter.generate(prompt)
            try:
                parsed = _parse_json(raw)
                block = StrategyBlock(**parsed)
                _validate_signal_ids_in_scope(block, insight)
                _validate_confidence_ceiling(block, insight.confidence)
                if attempt > 1:
                    logger.info(
                        "Strategy generation succeeded on attempt %d/%d",
                        attempt,
                        total_attempts,
                    )
                return block

            except (json.JSONDecodeError, TypeError) as exc:
                last_error = f"JSON parse error: {exc}"
                logger.warning(
                    "Strategy attempt %d/%d — %s",
                    attempt, total_attempts, last_error,
                )

            except Exception as exc:  # noqa: BLE001
                last_error = f"Validation error: {exc}"
                logger.warning(
                    "Strategy attempt %d/%d — %s",
                    attempt, total_attempts, last_error,
                )

        raise StrategyGeneratorError(
            attempts=total_attempts, last_error=last_error,
        )

    # -- prompt construction ------------------------------------------------

    def _build_prompt(self, insight: InsightBlock) -> str:
        parts: list[str] = [_SYSTEM_PROMPT, ""]

        # Flatten all signal references from the InsightBlock so the LLM
        # knows which SIG-<n> IDs are in scope.
        ref_table = _collect_signal_refs(insight)
        parts.append(
            "# PROVIDED DATA — SIGNAL REFERENCES\n"
            "These are the only signal IDs you may use:\n"
            f"```json\n{json.dumps(ref_table, indent=2)}\n```\n"
        )

        # InsightBlock summary for context
        parts.append(
            "# INSIGHT BLOCK SUMMARY\n"
            f"```json\n{_insight_summary_json(insight)}\n```\n"
        )

        # Quality
        quality = {
            "insight_confidence": round(insight.confidence, 4),
            "momentum_score": round(insight.momentum_score, 4),
            "opportunity_zones": len(insight.opportunity_zones),
            "risk_zones": len(insight.risk_zones),
        }
        if insight.confidence >= 0.8:
            quality["tone_directive"] = "definitive"
        elif insight.confidence >= 0.6:
            quality["tone_directive"] = "cautious"
        else:
            quality["tone_directive"] = "hedged"
        parts.append(
            f"# DATA QUALITY\n```json\n{json.dumps(quality, indent=2)}\n```\n"
        )

        # Schema
        parts.append(
            "# OUTPUT JSON SCHEMA\n"
            f"```json\n{json.dumps(StrategyBlock.model_json_schema(), indent=2)}\n```\n"
        )

        # Task
        parts.append(
            "# TASK\n"
            "Generate a single JSON object matching the schema above.  "
            "Ground every action in signal IDs.  No generic advice.  "
            "Exactly 3 short-term and 3 mid-term actions."
        )

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _parse_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def _collect_signal_refs(insight: InsightBlock) -> list[dict]:
    """Dedupe all signal references from the InsightBlock into a flat list."""
    seen: set[str] = set()
    refs: list[dict] = []
    for source in (
        *insight.emerging_signals,
        *insight.opportunity_zones,
        *insight.risk_zones,
    ):
        for ref in source.supporting_signals:
            if ref.signal_id not in seen:
                seen.add(ref.signal_id)
                refs.append({
                    "signal_id": ref.signal_id,
                    "metric_name": ref.metric_name,
                    "value": ref.value,
                    "unit": ref.unit,
                })
    return sorted(refs, key=lambda r: r["signal_id"])


def _insight_summary_json(insight: InsightBlock) -> str:
    """Compact JSON summary of the InsightBlock for prompt context."""
    summary = {
        "macro_summary": insight.macro_summary,
        "emerging_signals": [
            {"title": s.title, "relevance": s.relevance}
            for s in insight.emerging_signals
        ],
        "opportunity_zones": [
            {"title": z.title} for z in insight.opportunity_zones
        ],
        "risk_zones": [
            {"title": z.title} for z in insight.risk_zones
        ],
        "momentum_score": insight.momentum_score,
        "confidence": insight.confidence,
    }
    return json.dumps(summary, indent=2)


def _validate_signal_ids_in_scope(
    block: StrategyBlock,
    insight: InsightBlock,
) -> None:
    """Every SIG-<n> in the StrategyBlock must exist in the InsightBlock."""
    valid_ids: set[str] = set()
    for source in (
        *insight.emerging_signals,
        *insight.opportunity_zones,
        *insight.risk_zones,
    ):
        for ref in source.supporting_signals:
            valid_ids.add(ref.signal_id)

    all_refs: list[SignalReference] = []
    for act in block.short_term_actions:
        all_refs.extend(act.supporting_signals)
    for act in block.mid_term_actions:
        all_refs.extend(act.supporting_signals)
    all_refs.extend(block.competitive_angle.supporting_signals)
    for rm in block.risk_mitigation:
        all_refs.extend(rm.supporting_signals)

    bad_ids = {ref.signal_id for ref in all_refs if ref.signal_id not in valid_ids}
    if bad_ids:
        raise ValueError(
            f"Signal IDs not found in InsightBlock: {sorted(bad_ids)}. "
            f"Valid IDs: {sorted(valid_ids)}."
        )


def _validate_confidence_ceiling(
    block: StrategyBlock,
    insight_confidence: float,
) -> None:
    if block.confidence > insight_confidence + 1e-6:
        raise ValueError(
            f"StrategyBlock confidence ({block.confidence:.4f}) exceeds "
            f"InsightBlock confidence ceiling ({insight_confidence:.4f})."
        )
