"""
forecast/classifier.py

Classifies a regression slope into a human-readable trend label.
No forecasting logic, no I/O, no side effects.
"""

from __future__ import annotations


class TrendClassifier:
    """
    Maps a linear regression slope to a discrete trend category.

    Classification is relative to the magnitude of *average_value* so that
    a slope of 10 means something different for a KPI averaging 100 versus
    one averaging 10 000.

    Thresholds (class-level constants, easily overridden by subclasses):

        slope / average_value  |  label
        -----------------------|------------------
        > +5 %                 |  strong_uptrend
        > +1 %                 |  uptrend
        -1 % … +1 %            |  stable
        < -1 %                 |  downtrend
        < -5 %                 |  strong_downtrend
    """

    STRONG_UP_THRESHOLD:   float = 0.05   #  +5 %
    WEAK_UP_THRESHOLD:     float = 0.01   #  +1 %
    WEAK_DOWN_THRESHOLD:   float = -0.01  #  -1 %
    STRONG_DOWN_THRESHOLD: float = -0.05  #  -5 %

    def classify(self, slope: float, average_value: float) -> str:
        """
        Classify *slope* relative to *average_value*.

        Parameters
        ----------
        slope:
            Regression slope (units per period) produced by a forecast model.
        average_value:
            Mean of the historical KPI series used to normalise the slope.

        Returns
        -------
        str
            One of: ``"strong_uptrend"``, ``"uptrend"``, ``"stable"``,
            ``"downtrend"``, ``"strong_downtrend"``.

        Notes
        -----
        * When *average_value* is zero the raw slope is used directly as the
          ratio to avoid division by zero.
        * Thresholds are evaluated in order from most extreme to least
          extreme so boundary values resolve to the stronger label.
        """
        if average_value == 0.0:
            ratio: float = slope
        else:
            ratio = slope / abs(average_value)

        if ratio > self.STRONG_UP_THRESHOLD:
            return "strong_uptrend"
        if ratio > self.WEAK_UP_THRESHOLD:
            return "uptrend"
        if ratio < self.STRONG_DOWN_THRESHOLD:
            return "strong_downtrend"
        if ratio < self.WEAK_DOWN_THRESHOLD:
            return "downtrend"
        return "stable"
