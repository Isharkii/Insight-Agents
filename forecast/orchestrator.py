"""
forecast/orchestrator.py

Coordinates the forecast pipeline: regression → classification → persist.
Contains no forecasting math or classification logic.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, List

from sqlalchemy.orm import Session

from forecast.classifier import TrendClassifier
from forecast.regression import LinearRegressionForecast
from forecast.repository import ForecastRepository


class ForecastOrchestrator:
    """
    Thin coordinator that wires together the forecast pipeline.

    Each call to :meth:`generate_forecast` executes in order:

    1. Fit a linear regression model on *values*.
    2. Derive the average value of the series.
    3. Classify the slope into a trend label.
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
        self._model = LinearRegressionForecast()
        self._classifier = TrendClassifier()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_churn_acceleration(regression_result: dict[str, Any]) -> float | None:
        """
        Compute churn acceleration from three-step forecast values.

        Uses second finite difference:
            month_3 - 2 * month_2 + month_1

        Returns None when slope is unavailable.
        """
        slope = regression_result.get("slope")
        if slope is None:
            return None

        forecast = regression_result.get("forecast")
        if not isinstance(forecast, dict):
            return 0.0

        try:
            m1 = float(forecast["month_1"])
            m2 = float(forecast["month_2"])
            m3 = float(forecast["month_3"])
        except (KeyError, TypeError, ValueError):
            return 0.0

        return round(m3 - (2.0 * m2) + m1, 6)

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
            Must contain at least 2 points for a meaningful result.

        Returns
        -------
        dict
            On success::

                {
                    "metric_name":          str,
                    "slope":                float,
                    "trend":                str,
                    "forecast":             {"month_1": float, "month_2": float, "month_3": float},
                    "deviation_percentage": float,
                    "churn_acceleration":   float,
                }

            On insufficient data::

                {
                    "metric_name":          str,
                    "slope":                None,
                    "deviation_percentage": None,
                    "churn_acceleration":   None,
                    "error":                str,
                }
        """
        regression_result = self._model.forecast(values)
        churn_acceleration = self._compute_churn_acceleration(regression_result)

        # Propagate insufficient-data signal without saving.
        if regression_result.get("slope") is None:
            return {
                "metric_name": metric_name,
                "slope": None,
                "deviation_percentage": None,
                "churn_acceleration": None,
                "error": regression_result.get("error", "Insufficient data."),
            }

        average_value: float = sum(values) / len(values)
        slope: float = regression_result["slope"]

        trend: str = self._classifier.classify(
            slope=slope,
            average_value=average_value,
        )

        result: dict = {
            "metric_name": metric_name,
            "slope": slope,
            "trend": trend,
            "forecast": regression_result["forecast"],
            "deviation_percentage": regression_result["deviation_percentage"],
            "churn_acceleration": churn_acceleration,
        }

        repo = ForecastRepository(self._session)
        repo.save_forecast(
            entity_name=entity_name,
            metric_name=metric_name,
            period_end=datetime.now(timezone.utc),
            forecast_data=result,
        )

        return result
