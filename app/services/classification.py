"""
app/services/classification.py

Classification logic — maps a composite score (0–100) to a market
position label, confidence band, and relative position descriptor.

Separated from the scoring and ranking engines so that threshold
definitions can evolve independently of the math layer.

All computations are deterministic and LLM-free.

Default thresholds
------------------
    85–100  →  Market Leader
    70–84   →  Strong Performer
    50–69   →  Competitive
    30–49   →  Underperforming
     0–29   →  Critical Risk
"""

from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class ClassificationResult(BaseModel):
    """Immutable classification of a composite score."""

    model_config = ConfigDict(frozen=True)

    classification: str = Field(
        ...,
        description="Market position label (e.g. 'Strong Performer').",
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="The input composite score that was classified.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Distance from nearest threshold boundary, normalised to 0–1. "
            "1.0 = deep inside the band; values near 0 = borderline."
        ),
    )
    relative_position: str = Field(
        ...,
        description="Human-readable position descriptor (e.g. 'Top 25%').",
    )


# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------


class Tier:
    """A single classification tier with a lower bound and label."""

    __slots__ = ("lower", "label")

    def __init__(self, lower: float, label: str) -> None:
        self.lower = lower
        self.label = label


DEFAULT_TIERS: list[Tier] = [
    Tier(85.0, "Market Leader"),
    Tier(70.0, "Strong Performer"),
    Tier(50.0, "Competitive"),
    Tier(30.0, "Underperforming"),
    Tier(0.0, "Critical Risk"),
]

_POSITION_BRACKETS: list[tuple[float, str]] = [
    (90.0, "Top 10%"),
    (75.0, "Top 25%"),
    (50.0, "Top 50%"),
    (25.0, "Bottom 50%"),
    (0.0, "Bottom 25%"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_tiers(
    custom: Sequence[tuple[float, str]] | None,
) -> list[Tier]:
    """Build a sorted (descending by lower bound) tier list.

    Parameters
    ----------
    custom:
        Optional list of ``(lower_bound, label)`` tuples.
        If *None*, ``DEFAULT_TIERS`` is used.

    Each tuple defines a tier whose range extends from *lower_bound*
    up to the next tier's lower bound (or 100).
    """
    if custom is None:
        return list(DEFAULT_TIERS)
    tiers = [Tier(float(lb), str(lbl)) for lb, lbl in custom]
    tiers.sort(key=lambda t: t.lower, reverse=True)
    return tiers


def _classify_label(score: float, tiers: list[Tier]) -> tuple[str, float, float]:
    """Return (label, band_lower, band_upper) for *score*.

    Tiers must be sorted descending by lower bound.
    """
    for idx, tier in enumerate(tiers):
        if score >= tier.lower:
            band_lower = tier.lower
            band_upper = tiers[idx - 1].lower if idx > 0 else 100.0
            return tier.label, band_lower, band_upper
    # Fallback (should not happen if tiers include 0)
    last = tiers[-1]
    return last.label, last.lower, tiers[-2].lower if len(tiers) >= 2 else 100.0


def _confidence(score: float, band_lower: float, band_upper: float) -> float:
    """Compute classification confidence as distance from nearest boundary.

    Formula
    -------
    margin     = min(score − lower, upper − score)
    band_width = upper − lower
    confidence = margin / (band_width / 2)

    Clamped to [0, 1].  A score exactly at the centre of its band
    produces confidence 1.0; a score on the boundary produces ~0.0.
    """
    band_width = band_upper - band_lower
    if band_width <= 0:
        return 1.0
    margin = min(score - band_lower, band_upper - score)
    raw = margin / (band_width / 2.0)
    return round(max(0.0, min(1.0, raw)), 4)


def _relative_position(score: float) -> str:
    """Map score to a human-readable position bracket."""
    for threshold, label in _POSITION_BRACKETS:
        if score >= threshold:
            return label
    return _POSITION_BRACKETS[-1][1]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_score(
    score: float,
    *,
    tiers: Sequence[tuple[float, str]] | None = None,
) -> ClassificationResult:
    """Classify a composite score into a market position.

    Parameters
    ----------
    score:
        Composite score in [0, 100].
    tiers:
        Optional custom tier definitions as ``[(lower_bound, label), ...]``.
        Must cover the full 0–100 range (include a tier starting at 0).
        Defaults to the built-in five-tier scale.

    Returns
    -------
    ClassificationResult
        Classification label, confidence, and relative position.
    """
    score = max(0.0, min(100.0, float(score)))
    resolved = _resolve_tiers(tiers)
    label, band_lower, band_upper = _classify_label(score, resolved)
    conf = _confidence(score, band_lower, band_upper)
    position = _relative_position(score)

    return ClassificationResult(
        classification=label,
        score=round(score, 2),
        confidence=conf,
        relative_position=position,
    )
