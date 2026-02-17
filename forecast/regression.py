"""
forecast/regression.py

Lightweight linear regression forecasting using only standard Python / NumPy.
No sklearn, no statsmodels.
"""

from __future__ import annotations

from typing import List

from forecast.base import BaseForecastModel


class LinearRegressionForecast(BaseForecastModel):
    """
    Fits a simple OLS line to historical KPI values and projects 3 months ahead.

    The regression is computed analytically:

        m = cov(x, y) / var(x)
        b = mean(y) - m * mean(x)

    where x = [0, 1, ..., n-1] and y = the supplied values.

    Forecast points are extrapolated linearly from the last observed value:

        month_k = last_value + k * m   (k = 1, 2, 3)

    Deviation is the relative difference between the last observed value and
    the value the fitted line predicts at that position:

        deviation_pct = (actual_last - predicted_last) / predicted_last
    """

    # Minimum number of data points required for a meaningful fit.
    MIN_POINTS: int = 2

    def forecast(self, values: List[float]) -> dict:
        """
        Parameters
        ----------
        values:
            Monthly KPI values in chronological order (oldest first).

        Returns
        -------
        dict with keys:
            slope               – regression slope (m)
            intercept           – regression intercept (b)
            forecast            – dict with month_1 / month_2 / month_3
            deviation_percentage – (actual_last - predicted_last) / predicted_last
        """
        if len(values) < self.MIN_POINTS:
            return {
                "slope": None,
                "intercept": None,
                "forecast": {
                    "month_1": None,
                    "month_2": None,
                    "month_3": None,
                },
                "deviation_percentage": None,
                "error": (
                    f"Insufficient data: need at least {self.MIN_POINTS} points, "
                    f"got {len(values)}."
                ),
            }

        n = len(values)
        x = list(range(n))          # [0, 1, ..., n-1]
        y = list(values)            # copy; keep caller's list intact

        # --- descriptive statistics -----------------------------------------
        mean_x: float = sum(x) / n
        mean_y: float = sum(y) / n

        # covariance(x, y) and variance(x)  (population, not sample)
        cov_xy: float = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        var_x: float  = sum((x[i] - mean_x) ** 2 for i in range(n))

        if var_x == 0.0:
            # All x values are identical – degenerate case (n == 1 already
            # handled above, but guard here for safety).
            return {
                "slope": 0.0,
                "intercept": mean_y,
                "forecast": {
                    "month_1": mean_y,
                    "month_2": mean_y,
                    "month_3": mean_y,
                },
                "deviation_percentage": 0.0,
            }

        # --- regression coefficients ----------------------------------------
        slope: float     = cov_xy / var_x
        intercept: float = mean_y - slope * mean_x

        # --- 3-month forward projection from the last observed value ---------
        last_value: float = y[-1]
        forecast_1: float = last_value + 1 * slope
        forecast_2: float = last_value + 2 * slope
        forecast_3: float = last_value + 3 * slope

        # --- deviation at the last observed position -------------------------
        last_index: int      = n - 1
        predicted_last: float = slope * last_index + intercept

        if predicted_last == 0.0:
            deviation_pct: float = 0.0
        else:
            deviation_pct = (last_value - predicted_last) / predicted_last

        return {
            "slope": round(slope, 6),
            "intercept": round(intercept, 6),
            "forecast": {
                "month_1": round(forecast_1, 6),
                "month_2": round(forecast_2, 6),
                "month_3": round(forecast_3, 6),
            },
            "deviation_percentage": round(deviation_pct, 6),
        }
