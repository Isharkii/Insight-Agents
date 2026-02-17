"""
root_cause/base.py

Abstract base class for all root cause analysis engine implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseRootCauseEngine(ABC):
    """
    Contract for root cause analysis engine implementations.

    Subclasses receive KPI metrics, forecast projections, and risk
    assessments, and must return a plain dictionary describing the
    identified root causes and supporting evidence.

    No I/O, no logging, and no side effects are permitted inside
    :meth:`analyze`.
    """

    @abstractmethod
    def analyze(
        self,
        kpi_data: dict,
        forecast_data: dict,
        risk_data: dict,
    ) -> dict:
        """
        Identify root causes from the provided data inputs.

        Parameters
        ----------
        kpi_data:
            Computed KPI metrics and statistical summaries for the
            period under analysis.  Structure is defined by the
            upstream insight layer.

        forecast_data:
            Forecast results produced by a :class:`BaseForecastModel`
            implementation, including predicted values and model
            metadata.

        risk_data:
            Risk signals and scores produced by the risk evaluation
            layer.  May include anomaly flags, threshold breaches, and
            opportunity scores.

        Returns
        -------
        dict
            Root cause result.  Implementations must return at minimum:

            - ``"root_causes"``   – list of identified causal factors
            - ``"evidence"``      – supporting data points or signals
            - ``"impact"``        – assessed impact of each cause
            - ``"confidence"``    – confidence score in the range [0, 1]
            - ``"recommended_action"`` – suggested corrective measure
        """
