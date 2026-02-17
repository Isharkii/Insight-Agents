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
            Forecast result.  The exact keys are defined by each
            concrete subclass, but implementations are encouraged to
            include at minimum:

            - ``"forecast"`` – predicted next value or values
            - ``"model"``    – string identifier for the model used
        """
