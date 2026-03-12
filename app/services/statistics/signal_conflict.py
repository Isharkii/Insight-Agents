"""
app/services/statistics/signal_conflict.py

Semantic conflict detection for business KPI signals.

Unlike naive directional comparison (which only checks sign disagreement),
this module understands *business relationships* between metrics.  It knows
that "revenue up + churn up" is a conflict (revenue should not grow while
customers leave), while "revenue up + users up" is coherent.

Architecture
------------
1. A **conflict rule registry** encodes expected directional relationships
   between signal pairs.  Each rule specifies:
     - Two signal names (or families)
     - The expected relationship (``"same"`` or ``"opposite"``)
     - A severity weight

2. The **detector** receives a dict of signal deltas, classifies each as
   positive/negative/neutral, evaluates all applicable rules, and returns
   a structured conflict report.

3. The **confidence penalty** is computed from the weighted conflict count
   and fed directly into the pipeline's confidence scoring chain.

All logic is deterministic, config-driven, and LLM-independent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

_ZERO_GUARD = 1e-9


# ── Conflict rule registry ───────────────────────────────────────────────────

@dataclass(frozen=True)
class ConflictRule:
    """Defines an expected directional relationship between two signals.

    Parameters
    ----------
    signal_a, signal_b:
        Signal names (matched case-insensitively against the input dict).
    expected_relationship:
        ``"same"``     — both should move in the same direction
                         (e.g. revenue ↑ ↔ users ↑).
        ``"opposite"`` — they should move in opposite directions
                         (e.g. churn ↑ ↔ revenue ↓).
    severity:
        Weight of this conflict (0.0–1.0).  Higher = more damaging to
        confidence when violated.
    explanation:
        Human-readable description of why this pair conflicts.
    """

    signal_a: str
    signal_b: str
    expected_relationship: str  # "same" | "opposite"
    severity: float
    explanation: str


# ── Default rules ────────────────────────────────────────────────────────────
#
# These encode core business-domain knowledge.  Extend by appending to
# DEFAULT_CONFLICT_RULES or by passing a custom list to detect_conflicts().

DEFAULT_CONFLICT_RULES: list[ConflictRule] = [
    # ── Revenue × Churn ──────────────────────────────────────────────────
    ConflictRule(
        signal_a="revenue_growth_delta",
        signal_b="churn_delta",
        expected_relationship="opposite",
        severity=0.9,
        explanation=(
            "Revenue growth and churn should move in opposite directions. "
            "Rising revenue with rising churn suggests unsustainable growth "
            "driven by price increases rather than customer expansion."
        ),
    ),

    # ── Revenue × Conversion ─────────────────────────────────────────────
    ConflictRule(
        signal_a="revenue_growth_delta",
        signal_b="conversion_delta",
        expected_relationship="same",
        severity=0.7,
        explanation=(
            "Revenue and conversion rate should move together. "
            "Rising revenue with falling conversion suggests growth from "
            "existing customers rather than funnel health."
        ),
    ),

    # ── Traffic × Revenue ────────────────────────────────────────────────
    ConflictRule(
        signal_a="traffic_delta",
        signal_b="revenue_growth_delta",
        expected_relationship="same",
        severity=0.6,
        explanation=(
            "Traffic and revenue should generally co-move. "
            "Declining traffic with rising revenue may indicate price "
            "increases masking demand erosion."
        ),
    ),

    # ── Churn × Active Customers ─────────────────────────────────────────
    ConflictRule(
        signal_a="churn_delta",
        signal_b="active_customer_delta",
        expected_relationship="opposite",
        severity=0.8,
        explanation=(
            "Rising churn with stable or rising customer count is "
            "contradictory. Either churn measurement or customer "
            "count includes double-counting."
        ),
    ),

    # ── Growth Slope × Deviation ─────────────────────────────────────────
    ConflictRule(
        signal_a="slope",
        signal_b="deviation_percentage",
        expected_relationship="same",
        severity=0.5,
        explanation=(
            "Forecast slope direction should align with deviation. "
            "A positive slope with negative deviation suggests the "
            "model is under-fitting recent momentum."
        ),
    ),

    # ── Revenue × CAC ────────────────────────────────────────────────────
    ConflictRule(
        signal_a="revenue_growth_delta",
        signal_b="cac_delta",
        expected_relationship="opposite",
        severity=0.6,
        explanation=(
            "Rising revenue with rising CAC suggests growth is "
            "increasingly expensive and potentially unsustainable."
        ),
    ),

    # ── Conversion × Churn ───────────────────────────────────────────────
    ConflictRule(
        signal_a="conversion_delta",
        signal_b="churn_delta",
        expected_relationship="opposite",
        severity=0.7,
        explanation=(
            "Improving conversion with rising churn suggests the "
            "funnel attracts customers who don't retain — a quality "
            "versus quantity conflict."
        ),
    ),

    # ── LTV × Churn ──────────────────────────────────────────────────────
    ConflictRule(
        signal_a="ltv_delta",
        signal_b="churn_delta",
        expected_relationship="opposite",
        severity=0.8,
        explanation=(
            "LTV and churn must move in opposite directions. "
            "Rising LTV with rising churn is mathematically suspect."
        ),
    ),
]

# ── Signal family aliases ────────────────────────────────────────────────────
# Maps canonical signal names to alternative names found in different
# business type payloads.

_SIGNAL_ALIASES: dict[str, frozenset[str]] = {
    "revenue_growth_delta": frozenset({
        "revenue_growth_delta", "growth_rate", "revenue_delta",
        "mrr_delta", "total_revenue_delta",
    }),
    "churn_delta": frozenset({
        "churn_delta", "churn_rate_delta", "client_churn_delta",
        "churn_rate", "client_churn",
    }),
    "conversion_delta": frozenset({
        "conversion_delta", "conversion_rate_delta", "conversion_rate",
    }),
    "active_customer_delta": frozenset({
        "active_customer_delta", "active_customers_delta",
        "customer_count_delta",
    }),
    "traffic_delta": frozenset({
        "traffic_delta", "sessions_delta", "visits_delta",
    }),
    "cac_delta": frozenset({
        "cac_delta", "customer_acquisition_cost_delta",
    }),
    "ltv_delta": frozenset({
        "ltv_delta", "lifetime_value_delta", "client_ltv_delta",
    }),
    "slope": frozenset({"slope", "forecast_slope"}),
    "deviation_percentage": frozenset({
        "deviation_percentage", "deviation_pct",
    }),
}


def _build_reverse_alias_map() -> dict[str, str]:
    """Map every alias to its canonical name."""
    reverse: dict[str, str] = {}
    for canonical, aliases in _SIGNAL_ALIASES.items():
        for alias in aliases:
            reverse[alias.strip().lower()] = canonical
    return reverse


_REVERSE_ALIASES = _build_reverse_alias_map()


# ── Direction classifier ─────────────────────────────────────────────────────

def _classify_direction(value: float) -> str:
    """Classify a signal delta as positive, negative, or neutral."""
    if value > _ZERO_GUARD:
        return "positive"
    if value < -_ZERO_GUARD:
        return "negative"
    return "neutral"


def _directions_conflict(
    dir_a: str,
    dir_b: str,
    expected: str,
) -> bool:
    """Check if two directions violate the expected relationship.

    Parameters
    ----------
    dir_a, dir_b:
        ``"positive"``, ``"negative"``, or ``"neutral"``.
    expected:
        ``"same"``     — they should point the same way.
        ``"opposite"`` — they should point opposite ways.

    Returns True if the relationship is *violated*.
    """
    # Neutral signals cannot conflict — we lack directional evidence
    if dir_a == "neutral" or dir_b == "neutral":
        return False

    if expected == "same":
        # Same direction expected → conflict if they disagree
        return dir_a != dir_b
    if expected == "opposite":
        # Opposite direction expected → conflict if they agree
        return dir_a == dir_b

    return False


# ── Core detector ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SignalConflict:
    """A single detected contradiction between two signals."""

    rule: ConflictRule
    signal_a_value: float
    signal_b_value: float
    direction_a: str
    direction_b: str
    severity: float
    explanation: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "signal_a": self.rule.signal_a,
            "signal_b": self.rule.signal_b,
            "signal_a_value": round(self.signal_a_value, 6),
            "signal_b_value": round(self.signal_b_value, 6),
            "direction_a": self.direction_a,
            "direction_b": self.direction_b,
            "expected_relationship": self.rule.expected_relationship,
            "severity": round(self.severity, 6),
            "explanation": self.explanation,
        }


def detect_conflicts(
    signals: dict[str, float | None],
    *,
    rules: Sequence[ConflictRule] | None = None,
    confidence_penalty_per_unit: float = 0.12,
    max_total_penalty: float = 0.60,
) -> dict[str, Any]:
    """
    Detect semantic conflicts between business signals.

    Parameters
    ----------
    signals:
        Dict mapping signal names to delta values.
        Positive = growth/improvement direction.
        Negative = decline/deterioration direction.
        None values are ignored.
    rules:
        Conflict rules to evaluate.  Defaults to DEFAULT_CONFLICT_RULES.
    confidence_penalty_per_unit:
        Confidence penalty per unit of weighted severity.
    max_total_penalty:
        Maximum cumulative confidence penalty.

    Returns
    -------
    dict (JSON-compatible)
        conflicts          – list of detected conflict dicts
        conflict_count     – number of conflicts found
        total_severity     – sum of conflict severities
        confidence_penalty – penalty to apply to confidence score
        uncertainty_flag   – True if any high-severity conflict exists
        signal_directions  – classified directions for all input signals
        rules_evaluated    – number of rules that had both signals present
        rules_skipped      – number of rules missing one or both signals
        warnings           – human-readable conflict summaries
        status             – "conflicts_detected" | "clean" | "insufficient_signals"
    """
    if rules is None:
        rules = DEFAULT_CONFLICT_RULES

    # Canonicalise input signal names
    canonical_signals: dict[str, float] = {}
    for name, value in signals.items():
        if value is None:
            continue
        try:
            fval = float(value)
        except (TypeError, ValueError):
            continue
        key = _REVERSE_ALIASES.get(name.strip().lower(), name.strip().lower())
        canonical_signals[key] = fval

    if len(canonical_signals) < 2:
        return {
            "conflicts": [],
            "conflict_count": 0,
            "total_severity": 0.0,
            "confidence_penalty": 0.0,
            "uncertainty_flag": False,
            "signal_directions": {},
            "rules_evaluated": 0,
            "rules_skipped": len(rules),
            "warnings": [],
            "status": "insufficient_signals",
        }

    # Classify directions
    signal_directions: dict[str, str] = {
        name: _classify_direction(val)
        for name, val in canonical_signals.items()
    }

    # Evaluate rules
    detected: list[SignalConflict] = []
    rules_evaluated = 0
    rules_skipped = 0

    for rule in rules:
        a_key = rule.signal_a.strip().lower()
        b_key = rule.signal_b.strip().lower()

        a_val = canonical_signals.get(a_key)
        b_val = canonical_signals.get(b_key)

        if a_val is None or b_val is None:
            rules_skipped += 1
            continue

        rules_evaluated += 1
        dir_a = signal_directions[a_key]
        dir_b = signal_directions[b_key]

        if _directions_conflict(dir_a, dir_b, rule.expected_relationship):
            detected.append(SignalConflict(
                rule=rule,
                signal_a_value=a_val,
                signal_b_value=b_val,
                direction_a=dir_a,
                direction_b=dir_b,
                severity=rule.severity,
                explanation=rule.explanation,
            ))

    # Compute aggregate penalty
    total_severity = sum(c.severity for c in detected)
    raw_penalty = total_severity * confidence_penalty_per_unit
    confidence_penalty = min(max_total_penalty, raw_penalty)

    # Uncertainty flag: any conflict with severity >= 0.8
    uncertainty_flag = any(c.severity >= 0.8 for c in detected)

    # Generate warnings
    warnings: list[str] = []
    for c in detected:
        warnings.append(
            f"Conflict: {c.rule.signal_a} ({c.direction_a}) vs "
            f"{c.rule.signal_b} ({c.direction_b}) — "
            f"expected {c.rule.expected_relationship}, severity {c.severity:.1f}. "
            f"{c.explanation}"
        )

    if detected:
        status = "conflicts_detected"
    else:
        status = "clean"

    return {
        "conflicts": [c.as_dict() for c in detected],
        "conflict_count": len(detected),
        "total_severity": round(total_severity, 6),
        "confidence_penalty": round(confidence_penalty, 6),
        "uncertainty_flag": uncertainty_flag,
        "signal_directions": signal_directions,
        "rules_evaluated": rules_evaluated,
        "rules_skipped": rules_skipped,
        "warnings": warnings,
        "status": status,
    }


# ── Confidence integration ───────────────────────────────────────────────────

def apply_conflict_penalty(
    base_confidence: float,
    conflict_result: dict[str, Any],
    *,
    floor: float = 0.1,
) -> dict[str, Any]:
    """
    Apply conflict-derived penalty to a confidence score.

    Parameters
    ----------
    base_confidence:
        Input confidence score (0–1).
    conflict_result:
        Output of ``detect_conflicts()``.
    floor:
        Minimum confidence after penalty.

    Returns
    -------
    dict
        adjusted_confidence, penalty_applied, adjustment_reason,
        conflict_count, uncertainty_flag
    """
    penalty = float(conflict_result.get("confidence_penalty", 0.0))
    adjusted = max(floor, base_confidence - penalty)

    reason = "no_conflicts"
    if penalty > 0:
        reason = (
            f"{conflict_result['conflict_count']} signal conflict(s), "
            f"total severity {conflict_result['total_severity']:.2f}, "
            f"penalty {penalty:.3f}"
        )

    return {
        "adjusted_confidence": round(adjusted, 6),
        "base_confidence": round(base_confidence, 6),
        "penalty_applied": round(penalty, 6),
        "adjustment_reason": reason,
        "conflict_count": conflict_result.get("conflict_count", 0),
        "uncertainty_flag": conflict_result.get("uncertainty_flag", False),
    }
