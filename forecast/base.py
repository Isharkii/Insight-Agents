"""
forecast/base.py

Abstract base class for all forecast model implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseForecastModel(ABC):
    """
    Contract for forecast model implementations.

    Subclasses receive a sequence of historical numeric values and must
    return a plain dictionary describing the forecast result.

    No I/O, no logging, and no side effects are permitted inside
    :meth:`forecast`.
    """

    @abstractmethod
    def forecast(self, values: list[float]) -> dict:
        """
        Generate a forecast from *values* and return a result dictionary.

        Parameters
        ----------
        values:
            Ordered sequence of historical numeric observations, oldest
            first.  Must contain at least the minimum number of data
            points required by the concrete implementation.

        Returns
        -------
        dict
            Forecast result.  Implementations must include at minimum:

            - ``"forecast"``         – dict of predicted values (month_1, …)
            - ``"model"``            – string identifier for the model used
            - ``"status"``           – ``"ok"`` or ``"insufficient_data"``
            - ``"confidence_score"`` – float in [0, 1]
            - ``"tier"``             – data-depth tier label
            - ``"slope"``            – regression slope or None
            - ``"warnings"``         – list of diagnostic strings
        """
