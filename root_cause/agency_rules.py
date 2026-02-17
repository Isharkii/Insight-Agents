"""
root_cause/agency_rules.py

Deterministic, rule-based root cause engine for marketing agency metrics.
"""

from __future__ import annotations

from typing import List

from root_cause.base import BaseRootCauseEngine


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SEVERITY_BANDS: list[tuple[float, str]] = [
    (80.0, "critical"),
    (60.0, "high"),
    (30.0, "moderate"),
    (0.0,  "low"),
]


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


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class AgencyRootCauseEngine(BaseRootCauseEngine):
    """
    Rule-based root cause engine tuned for marketing agency KPI patterns.

    Applies a fixed, ordered set of deterministic rules against the
    supplied KPI, forecast, and risk dictionaries.  No statistical
    computation, no forecasting math, and no external I/O are performed
    here.

    Rules evaluated (in order)
    --------------------------
    1. Client retention issue   – client churn is rising.
    2. Underutilization problem – team utilization is falling.
    3. Productivity decline     – revenue per employee is falling.
    4. Capacity misalignment    – revenue and utilization both declining.
    5. Future revenue risk      – forecast slope is negative.
    6. High business risk       – risk score exceeds 70.

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
        Apply agency rules and return a structured root cause result.

        Parameters
        ----------
        kpi_data:
            May contain any of: ``revenue_growth_delta``,
            ``churn_delta``, ``utilization_delta``,
            ``revenue_per_employee_delta``.
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
        revenue_growth_delta: float        = float(kpi_data.get("revenue_growth_delta", 0.0))
        churn_delta: float                 = float(kpi_data.get("churn_delta", 0.0))
        utilization_delta: float           = float(kpi_data.get("utilization_delta", 0.0))
        revenue_per_employee_delta: float  = float(kpi_data.get("revenue_per_employee_delta", 0.0))

        slope: float                       = float(forecast_data.get("slope", 0.0))

        risk_score: float                  = float(risk_data.get("risk_score", 0.0))

        # ----------------------------------------------------------------
        # Rule evaluation
        # ----------------------------------------------------------------
        triggered: List[str] = []

        # Rule 1 – Client retention issue
        if churn_delta > 0:
            triggered.append("client_retention_issue")

        # Rule 2 – Underutilization problem
        if utilization_delta < 0:
            triggered.append("underutilization_problem")

        # Rule 3 – Productivity decline
        if revenue_per_employee_delta < 0:
            triggered.append("productivity_decline")

        # Rule 4 – Capacity misalignment
        if revenue_growth_delta < 0 and utilization_delta < 0:
            triggered.append("capacity_misalignment")

        # Rule 5 – Future revenue risk
        if slope < 0:
            triggered.append("future_revenue_risk")

        # Rule 6 – High business risk (additive, never primary alone)
        if risk_score > 70:
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
