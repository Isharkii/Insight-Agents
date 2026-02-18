"""
root_cause/ecommerce_rules.py

Deterministic, rule-based root cause engine for e-commerce business metrics.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import List

from root_cause.base import BaseRootCauseEngine


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BUSINESS_RULES_PATH = Path(__file__).resolve().parents[1] / "config" / "business_rules.yaml"


@lru_cache(maxsize=1)
def _load_business_rules() -> dict:
    try:
        raw = _BUSINESS_RULES_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _as_dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


_ROOT_CAUSE_RULES = _as_dict(_load_business_rules().get("root_cause"))
_SEVERITY_RULES = _as_dict(_ROOT_CAUSE_RULES.get("severity_thresholds"))
_ECOMMERCE_RULES = _as_dict(_ROOT_CAUSE_RULES.get("ecommerce"))

# Traffic is considered stable when its delta is within this band.
# The value is a percentage-point threshold; no arithmetic is applied.
_TRAFFIC_STABLE_THRESHOLD: float = _as_float(_ECOMMERCE_RULES.get("traffic_stable_threshold"), 5.0)

_SEVERITY_BANDS: list[tuple[float, str]] = [
    (_as_float(_SEVERITY_RULES.get("critical"), 80.0), "critical"),
    (_as_float(_SEVERITY_RULES.get("high"), 60.0), "high"),
    (_as_float(_SEVERITY_RULES.get("moderate"), 30.0), "moderate"),
    (0.0, "low"),
]

_HIGH_BUSINESS_RISK_THRESHOLD: float = _as_float(
    _ROOT_CAUSE_RULES.get("high_business_risk_threshold"),
    70.0,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _severity(risk_score: float) -> str:
    """Map a numeric risk score to a severity label.

    Parameters
    ----------
    risk_score:
        Numeric risk score, expected in the range [0, 100].

    Returns
    -------
    str
        One of ``"low"``, ``"moderate"``, ``"high"``, or ``"critical"``.
    """
    for threshold, label in _SEVERITY_BANDS:
        if risk_score > threshold:
            return label
    return "low"


def _traffic_stable(traffic_delta: float) -> bool:
    """Return True when traffic movement is within the stable band.

    Parameters
    ----------
    traffic_delta:
        Change in traffic volume expressed as a percentage-point delta.
        A value of ``0.0`` (the default when the key is absent) is
        treated as perfectly stable.

    Returns
    -------
    bool
    """
    return -_TRAFFIC_STABLE_THRESHOLD <= traffic_delta <= _TRAFFIC_STABLE_THRESHOLD


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class EcommerceRootCauseEngine(BaseRootCauseEngine):
    """
    Rule-based root cause engine tuned for e-commerce KPI patterns.

    Applies a fixed, ordered set of deterministic rules against the
    supplied KPI, forecast, and risk dictionaries.  No statistical
    computation, no forecasting math, and no external I/O are performed
    here.

    Rules evaluated (in order)
    --------------------------
    1. Conversion problem          – conversion falling with traffic stable.
    2. Pricing / product-mix issue – AOV falling while revenue is declining.
    3. Inefficient marketing spend – CAC rising with conversion falling.
    4. Retention problem           – repeat-purchase rate falling.
    5. Downward sales trend        – forecast slope is negative.
    6. High business risk          – risk score exceeds 70.

    The first matching rule sets ``primary_issue``; all subsequent
    matches are appended to ``contributing_factors``.
    """

    def analyze(
        self,
        kpi_data: dict,
        forecast_data: dict,
        risk_data: dict,
    ) -> dict:
        """
        Apply e-commerce rules and return a structured root cause result.

        Parameters
        ----------
        kpi_data:
            May contain any of: ``revenue_growth_delta``,
            ``conversion_delta``, ``aov_delta``, ``cac_delta``,
            ``repeat_purchase_delta``, ``traffic_delta``.
            Missing keys default to ``0.0``.

        forecast_data:
            May contain: ``slope``, ``deviation_percentage``.
            Missing keys default to ``0.0``.

        risk_data:
            May contain: ``risk_score``.
            Missing key defaults to ``0.0``.

        Returns
        -------
        dict
            ``{"primary_issue": str,
               "contributing_factors": List[str],
               "severity": str}``
        """
        # ----------------------------------------------------------------
        # Extract inputs with safe defaults
        # ----------------------------------------------------------------
        revenue_growth_delta: float   = float(kpi_data.get("revenue_growth_delta", 0.0))
        conversion_delta: float       = float(kpi_data.get("conversion_delta", 0.0))
        aov_delta: float              = float(kpi_data.get("aov_delta", 0.0))
        cac_delta: float              = float(kpi_data.get("cac_delta", 0.0))
        repeat_purchase_delta: float  = float(kpi_data.get("repeat_purchase_delta", 0.0))
        traffic_delta: float          = float(kpi_data.get("traffic_delta", 0.0))

        slope: float                  = float(forecast_data.get("slope", 0.0))

        risk_score: float             = float(risk_data.get("risk_score", 0.0))

        # ----------------------------------------------------------------
        # Rule evaluation
        # ----------------------------------------------------------------
        triggered: List[str] = []

        # Rule 1 – Conversion problem
        if conversion_delta < 0 and _traffic_stable(traffic_delta):
            triggered.append("conversion_problem")

        # Rule 2 – Pricing or product-mix issue
        if aov_delta < 0 and revenue_growth_delta < 0:
            triggered.append("pricing_or_product_mix_issue")

        # Rule 3 – Inefficient marketing spend
        if cac_delta > 0 and conversion_delta < 0:
            triggered.append("inefficient_marketing_spend")

        # Rule 4 – Retention problem
        if repeat_purchase_delta < 0:
            triggered.append("retention_problem")

        # Rule 5 – Downward sales trend
        if slope < 0:
            triggered.append("downward_sales_trend")

        # Rule 6 – High business risk (additive, never primary alone)
        if risk_score > _HIGH_BUSINESS_RISK_THRESHOLD:
            triggered.append("high_business_risk")

        # ----------------------------------------------------------------
        # Compose result
        # ----------------------------------------------------------------
        primary_issue: str              = triggered[0] if triggered else "no_issue_detected"
        contributing_factors: List[str] = triggered[1:] if len(triggered) > 1 else []

        return {
            "primary_issue":        primary_issue,
            "contributing_factors": contributing_factors,
            "severity":             _severity(risk_score),
        }
