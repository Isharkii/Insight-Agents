"""
agent/signal_envelope.py

Standardised SignalEnvelope contract for data and insight nodes.

Every node in the pipeline wraps its output in a SignalEnvelope that carries:
    - status      : "success" | "partial" | "skipped" | "failed"
    - payload     : the node's primary output data
    - warnings    : non-fatal issues the node encountered
    - errors      : fatal issues that prevented full computation
    - confidence_score : 0.0–1.0 expressing output reliability

The envelope is intentionally a plain dict for LangGraph state compatibility.
Typed helpers enforce structure at construction and extraction boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

EnvelopeStatus = Literal["success", "partial", "skipped", "failed"]

# Minimum signals a KPI stage must produce before risk/reasoning can proceed.
MINIMUM_KPI_SIGNALS: frozenset[str] = frozenset({
    "mrr",
    "churn_rate",
})

# Per-business-type minimum KPI signals.
MINIMUM_KPI_SIGNALS_BY_TYPE: dict[str, frozenset[str]] = {
    "saas": frozenset({"mrr", "churn_rate"}),
    "ecommerce": frozenset({"revenue", "conversion_rate"}),
    "agency": frozenset({"total_revenue", "client_churn"}),
}

# Optional KPI signals that enrich output but are not required.
OPTIONAL_KPI_SIGNALS: frozenset[str] = frozenset({
    "ltv",
    "growth_rate",
    "arpu",
    "conversion_rate",
    "utilization_rate",
    "revenue_per_employee",
    "purchase_frequency",
    "cac",
    "aov",
})

# Per-business-type optional KPI signals.
OPTIONAL_KPI_SIGNALS_BY_TYPE: dict[str, frozenset[str]] = {
    "saas": frozenset({"ltv", "growth_rate", "arpu"}),
    "ecommerce": frozenset({"aov", "cac", "purchase_frequency", "ltv", "growth_rate"}),
    "agency": frozenset({
        "retainer_revenue", "project_revenue", "utilization_rate",
        "revenue_per_employee", "client_ltv",
    }),
}


# ---------------------------------------------------------------------------
# Typed envelope dataclass (for construction / introspection)
# ---------------------------------------------------------------------------


@dataclass
class SignalEnvelope:
    """Typed wrapper around the dict-based envelope for safe construction."""

    status: EnvelopeStatus
    payload: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    confidence_score: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the plain dict expected by LangGraph state."""
        return {
            "status": self.status,
            "payload": self.payload,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "confidence_score": round(self.confidence_score, 4),
        }


# ---------------------------------------------------------------------------
# Construction helpers (drop-in replacements for node_result.py helpers)
# ---------------------------------------------------------------------------


def envelope_success(
    payload: dict[str, Any] | None,
    *,
    warnings: Sequence[str] = (),
    confidence_score: float = 1.0,
) -> dict[str, Any]:
    """Build a success envelope."""
    return SignalEnvelope(
        status="success",
        payload=payload,
        warnings=list(warnings),
        confidence_score=max(0.0, min(1.0, confidence_score)),
    ).to_dict()


def envelope_partial(
    payload: dict[str, Any] | None,
    *,
    warnings: Sequence[str] = (),
    errors: Sequence[str] = (),
    confidence_score: float = 0.5,
) -> dict[str, Any]:
    """Build a partial-success envelope (minimum signals present, optional missing)."""
    return SignalEnvelope(
        status="partial",
        payload=payload,
        warnings=list(warnings),
        errors=list(errors),
        confidence_score=max(0.0, min(1.0, confidence_score)),
    ).to_dict()


def envelope_skipped(
    reason: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a skipped envelope."""
    return SignalEnvelope(
        status="skipped",
        payload=payload,
        warnings=[reason] if reason else [],
        confidence_score=0.0,
    ).to_dict()


def envelope_failed(
    error: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a failed envelope."""
    return SignalEnvelope(
        status="failed",
        payload=payload,
        errors=[error] if error else [],
        confidence_score=0.0,
    ).to_dict()


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def envelope_status(value: Any) -> EnvelopeStatus:
    """Extract status from an envelope dict (backward-compatible with node_result)."""
    if isinstance(value, dict):
        status = value.get("status")
        if status in {"success", "partial", "skipped", "failed"}:
            return status
    return "failed"


def envelope_payload(value: Any) -> dict[str, Any] | None:
    """Extract payload from an envelope dict."""
    if isinstance(value, dict):
        status = value.get("status")
        payload = value.get("payload")
        if status in {"success", "partial", "skipped", "failed"}:
            return payload if isinstance(payload, dict) else None
        return value
    return None


def envelope_warnings(value: Any) -> list[str]:
    """Extract warnings from an envelope dict."""
    if isinstance(value, dict):
        warnings = value.get("warnings")
        if isinstance(warnings, list):
            return [str(w) for w in warnings]
    return []


def envelope_errors(value: Any) -> list[str]:
    """Extract errors from an envelope dict."""
    if isinstance(value, dict):
        errors = value.get("errors")
        if isinstance(errors, list):
            return [str(e) for e in errors]
    return []


def envelope_confidence(value: Any) -> float:
    """Extract confidence_score from an envelope dict."""
    if isinstance(value, dict):
        score = value.get("confidence_score")
        if isinstance(score, (int, float)):
            return float(score)
    return 0.0


# ---------------------------------------------------------------------------
# Partial-result evaluation
# ---------------------------------------------------------------------------


def has_minimum_signals(
    computed_kpis: dict[str, Any],
    *,
    required: frozenset[str] = MINIMUM_KPI_SIGNALS,
) -> tuple[bool, list[str]]:
    """
    Check whether computed KPIs contain the minimum required signals.

    Returns (ok, missing_list).
    """
    missing: list[str] = []
    for signal in required:
        entry = computed_kpis.get(signal)
        if entry is None:
            missing.append(signal)
            continue
        # Support both raw values and {value: ..., error: ...} dicts
        if isinstance(entry, dict):
            if entry.get("value") is None or entry.get("error") is not None:
                missing.append(signal)
        elif entry is None:
            missing.append(signal)
    return len(missing) == 0, missing


def classify_kpi_completeness(
    computed_kpis: dict[str, Any],
) -> tuple[EnvelopeStatus, list[str], list[str], float]:
    """
    Classify KPI output completeness.

    Returns:
        (status, warnings, errors, confidence_score)

    Rules:
        - All minimum + optional present → "success", confidence=1.0
        - All minimum present, some optional missing → "partial", confidence=0.6–0.9
        - Some minimum missing → "failed"
    """
    min_ok, min_missing = has_minimum_signals(computed_kpis)

    if not min_ok:
        return (
            "failed",
            [],
            [f"Missing required signal: {s}" for s in min_missing],
            0.0,
        )

    warnings: list[str] = []
    optional_present = 0
    optional_total = 0
    for signal in OPTIONAL_KPI_SIGNALS:
        entry = computed_kpis.get(signal)
        if entry is None:
            continue  # not expected for this business type
        optional_total += 1
        if isinstance(entry, dict):
            if entry.get("value") is not None and entry.get("error") is None:
                optional_present += 1
            else:
                warnings.append(
                    f"Optional signal '{signal}' could not be computed: "
                    f"{entry.get('error', 'value is None')}"
                )
        else:
            optional_present += 1

    if optional_total == 0:
        return "success", warnings, [], 1.0

    ratio = optional_present / optional_total
    confidence = round(0.6 + 0.4 * ratio, 4)

    if ratio >= 1.0:
        return "success", warnings, [], confidence
    return "partial", warnings, [], confidence


def classify_kpi_completeness_for_type(
    computed_kpis: dict[str, Any],
    business_type: str,
) -> tuple[EnvelopeStatus, list[str], list[str], float]:
    """
    Business-type-aware KPI completeness classification.

    Uses per-type minimum and optional signal registries.  Falls back to
    the generic :func:`classify_kpi_completeness` if the business type is
    not registered.
    """
    required = MINIMUM_KPI_SIGNALS_BY_TYPE.get(business_type)
    optional = OPTIONAL_KPI_SIGNALS_BY_TYPE.get(business_type)

    if required is None:
        return classify_kpi_completeness(computed_kpis)

    min_ok, min_missing = has_minimum_signals(computed_kpis, required=required)
    if not min_ok:
        return (
            "failed",
            [],
            [f"Missing required signal: {s}" for s in min_missing],
            0.0,
        )

    if optional is None:
        return "success", [], [], 1.0

    warnings: list[str] = []
    optional_present = 0
    optional_total = 0
    for signal in optional:
        entry = computed_kpis.get(signal)
        if entry is None:
            continue
        optional_total += 1
        if isinstance(entry, dict):
            if entry.get("value") is not None and entry.get("error") is None:
                optional_present += 1
            else:
                warnings.append(
                    f"Optional signal '{signal}' could not be computed: "
                    f"{entry.get('error', 'value is None')}"
                )
        else:
            optional_present += 1

    if optional_total == 0:
        return "success", warnings, [], 1.0

    ratio = optional_present / optional_total
    confidence = round(0.6 + 0.4 * ratio, 4)

    if ratio >= 1.0:
        return "success", warnings, [], confidence
    return "partial", warnings, [], confidence
