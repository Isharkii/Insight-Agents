"""
root_cause/saas_rules.py

Deterministic, rule-based root cause engine for SaaS business metrics.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import List

from root_cause.base import BaseRootCauseEngine


# ---------------------------------------------------------------------------
# Severity thresholds
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


def _require_float(payload: dict, key: str, payload_name: str) -> float:
    if key not in payload:
        raise ValueError(f"Missing required {payload_name} signal '{key}'.")
    value = payload[key]
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid {payload_name} signal '{key}': expected numeric value."
        ) from exc


_ROOT_CAUSE_RULES = _as_dict(_load_business_rules().get("root_cause"))
_SEVERITY_RULES = _as_dict(_ROOT_CAUSE_RULES.get("severity_thresholds"))

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
            Required keys: ``revenue_growth_delta``, ``churn_delta``,
            ``cac_delta``, ``ltv_delta``.

        forecast_data:
            Required keys: ``slope``, ``deviation_percentage``.

        risk_data:
            Required key: ``risk_score``.

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
        if not isinstance(kpi_data, dict):
            raise ValueError("kpi_data must be a dict.")
        if not isinstance(forecast_data, dict):
            raise ValueError("forecast_data must be a dict.")
        if not isinstance(risk_data, dict):
            raise ValueError("risk_data must be a dict.")

        revenue_growth_delta: float = _require_float(
            kpi_data, "revenue_growth_delta", "kpi_data"
        )
        churn_delta: float = _require_float(kpi_data, "churn_delta", "kpi_data")
        cac_delta: float = _require_float(kpi_data, "cac_delta", "kpi_data")
        ltv_delta: float = _require_float(kpi_data, "ltv_delta", "kpi_data")

        slope: float = _require_float(forecast_data, "slope", "forecast_data")
        deviation_pct: float = _require_float(
            forecast_data, "deviation_percentage", "forecast_data"
        )

        risk_score: float = _require_float(risk_data, "risk_score", "risk_data")

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
        if risk_score > _HIGH_BUSINESS_RISK_THRESHOLD:
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
