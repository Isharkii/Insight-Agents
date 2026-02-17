"""
risk/scoring.py

Business Risk Index model implementing BaseRiskModel.
Computes a weighted, normalized risk score from KPI and forecast inputs.
"""

from risk.base import BaseRiskModel
from risk.normalizer import RiskNormalizer


class BusinessRiskModel(BaseRiskModel):
    """Weighted risk scoring model for business health assessment.

    Combines KPI deltas, forecast metrics, and acceleration signals
    into a single normalized risk score on a 0–100 scale.

    All weights must sum to 1.0. Each signal is independently
    normalized before weighting to ensure comparability.
    """

    # Scoring weights — must sum to 1.0
    REVENUE_WEIGHT: float = 0.25
    CHURN_WEIGHT: float = 0.25
    FORECAST_WEIGHT: float = 0.20
    DEVIATION_WEIGHT: float = 0.15
    ACCELERATION_WEIGHT: float = 0.15

    # Upper bounds used for positive-range normalization
    MAX_DEVIATION_PCT: float = 1.0
    MAX_SLOPE: float = 1.0
    MAX_CHURN_ACCELERATION: float = 1.0

    def __init__(self) -> None:
        """Initialize the model with a shared RiskNormalizer instance."""
        self._normalizer = RiskNormalizer()

    def compute(self, inputs: dict) -> float:
        """Compute a Business Risk Index score from the given input signals.

        Each input is normalized to [0, 1] where 1 represents maximum
        risk contribution. Normalized values are multiplied by their
        respective weights and summed, then scaled to a 0–100 range.

        Missing keys default to 0.0 (no risk contribution from that signal).

        Signal directionality:
            - revenue_growth_delta: negative values → higher risk
            - churn_delta: positive values → higher risk
            - slope: negative values → higher risk
            - deviation_percentage: higher values → higher risk
            - churn_acceleration: higher values → higher risk
            - conversion_delta: present in inputs schema, not weighted here

        Args:
            inputs: Dictionary with any of the following optional keys:
                - revenue_growth_delta (float): Delta in [-1, +1].
                - churn_delta (float): Delta in [-1, +1].
                - conversion_delta (float): Available but not weighted.
                - deviation_percentage (float): Forecast deviation >= 0.
                - slope (float): Forecast trend slope, any sign.
                - churn_acceleration (float): Acceleration signal >= 0.

        Returns:
            A float in [0.0, 100.0] representing the Business Risk Index.
            Higher values indicate greater business risk.
        """
        n = self._normalizer

        revenue_growth_delta: float = inputs.get("revenue_growth_delta", 0.0)
        churn_delta: float = inputs.get("churn_delta", 0.0)
        deviation_percentage: float = inputs.get("deviation_percentage", 0.0)
        slope: float = inputs.get("slope", 0.0)
        churn_acceleration: float = inputs.get("churn_acceleration", 0.0)

        # Normalize each signal to [0, 1] risk contribution.
        # Negative revenue growth → invert sign before normalizing.
        rev_risk: float = n.normalize_percentage(-revenue_growth_delta)

        # Positive churn delta → directly maps to higher risk.
        churn_risk: float = n.normalize_percentage(churn_delta)

        # Negative slope → invert sign; normalize_positive clamps negatives to 0.
        forecast_risk: float = n.normalize_positive(-slope, self.MAX_SLOPE)

        # Deviation and acceleration are inherently positive risk signals.
        deviation_risk: float = n.normalize_positive(deviation_percentage, self.MAX_DEVIATION_PCT)
        acc_risk: float = n.normalize_positive(churn_acceleration, self.MAX_CHURN_ACCELERATION)

        weighted_sum: float = (
            rev_risk * self.REVENUE_WEIGHT
            + churn_risk * self.CHURN_WEIGHT
            + forecast_risk * self.FORECAST_WEIGHT
            + deviation_risk * self.DEVIATION_WEIGHT
            + acc_risk * self.ACCELERATION_WEIGHT
        )

        return float(round(n.clamp(weighted_sum * 100.0, 0.0, 100.0)))
