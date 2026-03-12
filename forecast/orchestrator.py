"""
forecast/orchestrator.py

Coordinates the forecast pipeline: robust forecast → classification → persist.
Contains no forecasting math or classification logic.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, List

from sqlalchemy.orm import Session

from forecast.classifier import TrendClassifier
from forecast.repository import ForecastRepository
from forecast.robust_forecast import RobustForecast


class ForecastOrchestrator:
    """
    Thin coordinator that wires together the forecast pipeline.

    Each call to :meth:`generate_forecast` executes in order:

    1. Run the robust tiered forecast model on *values*.
    2. Extract trend label from the model output (or classify via slope).
    3. Compute churn acceleration from the 3-month projection.
    4. Assemble the canonical output dictionary.
    5. Persist the result via :class:`ForecastRepository`.

    The caller is responsible for committing the enclosing database
    transaction; this class never calls ``session.commit()``.

    Parameters
    ----------
    session:
        An active :class:`sqlalchemy.orm.Session` passed through to the
        repository layer.
    """

    def __init__(self, session: Session) -> None:
        self._session = session
        self._model = RobustForecast()
        self._classifier = TrendClassifier()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_churn_acceleration(forecast_result: dict[str, Any]) -> float | None:
        """
        Compute churn acceleration from three-step forecast values.

        Uses second finite difference:
            month_3 - 2 * month_2 + month_1

        Returns None when slope is unavailable.
        """
        slope = forecast_result.get("slope")
        if slope is None:
            return None

        forecast = forecast_result.get("forecast")
        if not isinstance(forecast, dict):
            return 0.0

        try:
            m1 = float(forecast["month_1"])
            m2 = float(forecast["month_2"])
            m3 = float(forecast["month_3"])
        except (KeyError, TypeError, ValueError):
            return 0.0

        return round(m3 - (2.0 * m2) + m1, 6)

    @staticmethod
    def _insufficient_data_result(metric_name: str, message: str) -> dict[str, Any]:
        """Return a structured non-throwing payload for insufficient input history."""
        return {
            "metric_name": metric_name,
            "status": "insufficient_data",
            "forecast_available": False,
            "slope": None,
            "deviation_percentage": None,
            "churn_acceleration": None,
            "confidence_score": 0.0,
            "tier": "insufficient",
            "error": message,
        }

    def generate_forecast(
        self,
        entity_name: str,
        metric_name: str,
        values: List[float],
    ) -> dict:
        """
        Run the full forecast pipeline for one entity / metric pair.

        Parameters
        ----------
        entity_name:
            Logical owner of the metric (client, product, region …).
        metric_name:
            KPI identifier (e.g. ``"monthly_revenue"``).
        values:
            Monthly KPI values in chronological order, oldest first.

        Returns
        -------
        dict
            On success::

                {
                    "metric_name":          str,
                    "status":               "ok",
                    "forecast_available":   True,
                    "tier":                 str,
                    "confidence_score":     float,
                    "slope":                float | None,
                    "trend":                str,
                    "forecast":             {"month_1": float, …},
                    "deviation_percentage": float | None,
                    "churn_acceleration":   float | None,
                    "data_quality":         dict,
                    "regression":           dict | None,
                    "warnings":             list[str],
                }

            On insufficient data::

                {
                    "metric_name":          str,
                    "status":               "insufficient_data",
                    "forecast_available":   False,
                    "slope":                None,
                    "confidence_score":     0.0,
                    "tier":                 "insufficient",
                    "error":                str,
                }
        """
        forecast_result = self._model.forecast(values)
        churn_acceleration = self._compute_churn_acceleration(forecast_result)

        # Propagate insufficient-data signal without saving.
        if forecast_result.get("status") == "insufficient_data":
            reason = ""
            diagnostics = forecast_result.get("diagnostics")
            if isinstance(diagnostics, dict):
                reason = diagnostics.get("reason", "")
            return self._insufficient_data_result(
                metric_name=metric_name,
                message=reason or "Insufficient data.",
            )

        # Use trend from the robust model if available, else classify via slope
        trend_info = forecast_result.get("trend") or {}
        trend_label = trend_info.get("label") if isinstance(trend_info, dict) else None

        if not trend_label or trend_label == "inconclusive":
            slope = forecast_result.get("slope")
            if slope is not None:
                average_value = sum(values) / len(values) if values else 0.0
                trend_label = self._classifier.classify(
                    slope=slope,
                    average_value=average_value,
                )
            else:
                trend_label = "inconclusive"

        result: dict[str, Any] = {
            "metric_name": metric_name,
            "status": "ok",
            "forecast_available": True,
            "tier": forecast_result.get("tier", "unknown"),
            "confidence_score": forecast_result.get("confidence_score", 0.0),
            "slope": forecast_result.get("slope"),
            "trend": trend_label,
            "forecast": forecast_result.get("forecast", {}),
            "deviation_percentage": forecast_result.get("deviation_percentage"),
            "churn_acceleration": churn_acceleration,
            "data_quality": forecast_result.get("data_quality", {}),
            "regression": forecast_result.get("regression"),
            "input_points": forecast_result.get("input_points", len(values)),
            "warnings": forecast_result.get("warnings", []),
        }

        repo = ForecastRepository(self._session)
        repo.save_forecast(
            entity_name=entity_name,
            metric_name=metric_name,
            period_end=datetime.now(timezone.utc),
            forecast_data=result,
        )

        return result
