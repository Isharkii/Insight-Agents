"""
risk/orchestrator.py

Orchestrates risk score generation by coordinating BusinessRiskModel
and RiskRepository. Contains no scoring, KPI, or forecasting logic.
"""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from risk.repository import RiskRepository
from risk.scoring import BusinessRiskModel


# ---------------------------------------------------------------------------
# Risk level classification - thresholds are inclusive upper bounds
# ---------------------------------------------------------------------------

_RISK_THRESHOLDS: tuple[tuple[int, str], ...] = (
    (30, "low"),
    (60, "moderate"),
    (80, "high"),
    (100, "critical"),
)

_FORECAST_DEPENDENT_SIGNALS: tuple[str, ...] = (
    "deviation_percentage",
    "slope",
    "churn_acceleration",
)


def _to_float(value: Any, default: float = 0.0) -> float:
    """Best-effort numeric coercion for resilient risk input handling."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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

    def _is_forecast_available(self, forecast_data: dict[str, Any]) -> bool:
        """Determine whether forecast-derived signals are usable."""
        if not isinstance(forecast_data, dict):
            return False

        if forecast_data.get("forecast_available") is False:
            return False

        status_value = str(forecast_data.get("status", "")).strip().lower()
        if status_value == "insufficient_data":
            return False

        if forecast_data.get("forecast_available") is True:
            return True

        present_values = [
            forecast_data.get(signal)
            for signal in _FORECAST_DEPENDENT_SIGNALS
            if signal in forecast_data
        ]
        if not present_values:
            return False

        return not all(value is None for value in present_values)

    def _active_weight_for_available_signals(self, forecast_available: bool) -> float:
        """Return sum of configured weights for currently active signals."""
        active_weight = self._model.REVENUE_WEIGHT + self._model.CHURN_WEIGHT
        if forecast_available:
            active_weight += (
                self._model.FORECAST_WEIGHT
                + self._model.DEVIATION_WEIGHT
                + self._model.ACCELERATION_WEIGHT
            )
        return max(0.0, float(active_weight))

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
        kpi_payload: dict[str, Any] = kpi_data if isinstance(kpi_data, dict) else {}
        forecast_payload: dict[str, Any] = (
            forecast_data if isinstance(forecast_data, dict) else {}
        )
        forecast_available = self._is_forecast_available(forecast_payload)

        inputs: dict[str, float] = {
            "revenue_growth_delta": _to_float(kpi_payload.get("revenue_growth_delta", 0.0)),
            "churn_delta": _to_float(kpi_payload.get("churn_delta", 0.0)),
            "conversion_delta": _to_float(kpi_payload.get("conversion_delta", 0.0)),
            "deviation_percentage": (
                _to_float(forecast_payload.get("deviation_percentage", 0.0))
                if forecast_available
                else 0.0
            ),
            "slope": (
                _to_float(forecast_payload.get("slope", 0.0))
                if forecast_available
                else 0.0
            ),
            "churn_acceleration": (
                _to_float(forecast_payload.get("churn_acceleration", 0.0))
                if forecast_available
                else 0.0
            ),
        }

        raw_score: float = self._model.compute(inputs)
        active_weight: float = self._active_weight_for_available_signals(
            forecast_available=forecast_available,
        )
        if forecast_available:
            adjusted_score = raw_score
            reweight_factor = 1.0
        elif active_weight > 0.0:
            reweight_factor = 1.0 / active_weight
            adjusted_score = raw_score * reweight_factor
        else:
            reweight_factor = 0.0
            adjusted_score = 0.0

        risk_score: int = int(max(0.0, min(100.0, adjusted_score)))
        risk_level: str = _classify(risk_score)

        metadata: dict[str, Any] = {
            **inputs,
            "forecast_available": forecast_available,
            "forecast_signals_skipped": not forecast_available,
            "active_weight": round(active_weight, 6),
            "reweight_factor": round(reweight_factor, 6),
        }
        if not forecast_available:
            metadata["skipped_signals"] = list(_FORECAST_DEPENDENT_SIGNALS)

        self._repository.save_risk_score(
            session=self._session,
            entity_name=entity_name,
            period_end=datetime.now(timezone.utc),
            risk_score=risk_score,
            risk_metadata=metadata,
        )

        return {
            "entity_name": entity_name,
            "risk_score": risk_score,
            "risk_level": risk_level,
        }
