"""
risk/orchestrator.py

Orchestrates risk score generation by coordinating BusinessRiskModel
and RiskRepository. Contains no scoring, KPI, or forecasting logic.
"""

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from risk.scoring import BusinessRiskModel
from risk.repository import RiskRepository


# ---------------------------------------------------------------------------
# Risk level classification — thresholds are inclusive upper bounds
# ---------------------------------------------------------------------------

_RISK_THRESHOLDS: tuple[tuple[int, str], ...] = (
    (30,  "low"),
    (60,  "moderate"),
    (80,  "high"),
    (100, "critical"),
)


def _classify(score: int) -> str:
    """Map an integer score in [0, 100] to a risk level label.

    Args:
        score: Integer risk index.

    Returns:
        One of "low", "moderate", "high", or "critical".
    """
    for threshold, label in _RISK_THRESHOLDS:
        if score <= threshold:
            return label
    return "critical"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class RiskOrchestrator:
    """Coordinates risk score generation, classification, and persistence.

    Accepts a SQLAlchemy Session at construction time so the caller
    retains full control over transaction lifecycle (commit / rollback).
    Delegates computation to BusinessRiskModel and persistence to
    RiskRepository. Contains no scoring or domain-specific math.
    """

    def __init__(self, session: Session) -> None:
        """
        Args:
            session: Active SQLAlchemy session passed to the repository.
        """
        self._session = session
        self._model = BusinessRiskModel()
        self._repository = RiskRepository()

    def generate_risk_score(
        self,
        entity_name: str,
        kpi_data: dict,
        forecast_data: dict,
    ) -> dict:
        """Generate, persist, and return a classified risk score.

        Extracts input signals from kpi_data and forecast_data, computes
        a risk index via BusinessRiskModel, saves the record via
        RiskRepository, and returns a structured result.

        Missing keys in either input dict default to 0.0.

        Args:
            entity_name: Identifier for the business entity being scored.
            kpi_data: KPI delta signals. Expected keys:
                - revenue_growth_delta (float)
                - churn_delta (float)
                - conversion_delta (float)
            forecast_data: Forecast signals. Expected keys:
                - deviation_percentage (float)
                - slope (float)
                - churn_acceleration (float)

        Returns:
            {
                "entity_name": str,
                "risk_score": int,
                "risk_level": str  # "low" | "moderate" | "high" | "critical"
            }
        """
        inputs: dict = {
            "revenue_growth_delta": kpi_data.get("revenue_growth_delta", 0.0),
            "churn_delta":          kpi_data.get("churn_delta", 0.0),
            "conversion_delta":     kpi_data.get("conversion_delta", 0.0),
            "deviation_percentage": forecast_data.get("deviation_percentage", 0.0),
            "slope":                forecast_data.get("slope", 0.0),
            "churn_acceleration":   forecast_data.get("churn_acceleration", 0.0),
        }

        raw_score: float = self._model.compute(inputs)
        risk_score: int = int(raw_score)
        risk_level: str = _classify(risk_score)

        self._repository.save_risk_score(
            session=self._session,
            entity_name=entity_name,
            period_end=datetime.now(timezone.utc),
            risk_score=risk_score,
            risk_metadata=inputs,
        )

        return {
            "entity_name": entity_name,
            "risk_score": risk_score,
            "risk_level": risk_level,
        }
