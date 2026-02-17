"""
root_cause/saas_rules.py

Deterministic, rule-based root cause engine for SaaS business metrics.
"""

from __future__ import annotations

from typing import List

from root_cause.base import BaseRootCauseEngine


# ---------------------------------------------------------------------------
# Severity thresholds
# ---------------------------------------------------------------------------

_SEVERITY_BANDS: list[tuple[float, str]] = [
    (80.0, "critical"),
    (60.0, "high"),
    (30.0, "moderate"),
    (0.0,  "low"),
]


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


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SaaSRootCauseEngine(BaseRootCauseEngine):
    """
    Rule-based root cause engine tuned for SaaS KPI patterns.

    Applies a fixed, ordered set of deterministic rules against the
    supplied KPI, forecast, and risk dictionaries.  No statistical
    computation, no forecasting math, and no external I/O are performed
    here.

    Rules evaluated (in order)
    --------------------------
    1. Retention issue       – churn rising while revenue falling.
    2. Acquisition inefficiency – CAC rising with flat/negative revenue.
    3. Customer value decline – churn rising with LTV falling.
    4. Negative growth trend – forecast slope and deviation both negative.
    5. High business risk    – risk score exceeds 70.

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
        Apply SaaS rules and return a structured root cause result.

        Parameters
        ----------
        kpi_data:
            May contain any of: ``revenue_growth_delta``,
            ``churn_delta``, ``cac_delta``, ``ltv_delta``.
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
        revenue_growth_delta: float = float(kpi_data.get("revenue_growth_delta", 0.0))
        churn_delta: float          = float(kpi_data.get("churn_delta", 0.0))
        cac_delta: float            = float(kpi_data.get("cac_delta", 0.0))
        ltv_delta: float            = float(kpi_data.get("ltv_delta", 0.0))

        slope: float                = float(forecast_data.get("slope", 0.0))
        deviation_pct: float        = float(forecast_data.get("deviation_percentage", 0.0))

        risk_score: float           = float(risk_data.get("risk_score", 0.0))

        # ----------------------------------------------------------------
        # Rule evaluation
        # ----------------------------------------------------------------
        triggered: List[str] = []

        # Rule 1 – Retention issue
        if churn_delta > 0 and revenue_growth_delta < 0:
            triggered.append("retention_issue")

        # Rule 2 – Acquisition inefficiency
        if cac_delta > 0 and revenue_growth_delta <= 0:
            triggered.append("acquisition_inefficiency")

        # Rule 3 – Customer value decline
        if churn_delta > 0 and ltv_delta < 0:
            triggered.append("customer_value_decline")

        # Rule 4 – Negative growth trend
        if slope < 0 and deviation_pct < 0:
            triggered.append("negative_growth_trend")

        # Rule 5 – High business risk (additive, never primary alone)
        if risk_score > 70:
            triggered.append("high_business_risk")

        # ----------------------------------------------------------------
        # Compose result
        # ----------------------------------------------------------------
        primary_issue: str          = triggered[0] if triggered else "no_issue_detected"
        contributing_factors: List[str] = triggered[1:] if len(triggered) > 1 else []

        return {
            "primary_issue":         primary_issue,
            "contributing_factors":  contributing_factors,
            "severity":              _severity(risk_score),
        }
