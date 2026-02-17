"""
risk/normalizer.py

Deterministic metric normalization utilities for risk scoring inputs.
"""


class RiskNormalizer:
    """Provides stateless normalization methods for risk input metrics.

    All methods are deterministic and produce bounded float outputs.
    No external dependencies, state, or side effects.
    """

    def normalize_percentage(self, value: float) -> float:
        """Convert a value in the range [-1, +1] to a [0, 1] scale.

        Args:
            value: A float assumed to be between -1 and +1 inclusive.

        Returns:
            A float in the range [0, 1].
        """
        return (value + 1.0) / 2.0

    def normalize_positive(self, value: float, max_expected: float) -> float:
        """Normalize a positive value against an expected maximum.

        Computes value / max_expected and clamps the result to [0, 1].

        Args:
            value: The raw input value.
            max_expected: The expected upper bound for normalization.

        Returns:
            A float in the range [0, 1].

        Raises:
            ValueError: If max_expected is zero.
        """
        if max_expected == 0:
            raise ValueError("max_expected must not be zero.")
        return self.clamp(value / max_expected, 0.0, 1.0)

    def clamp(self, value: float, min_value: float, max_value: float) -> float:
        """Clamp a value to the specified [min_value, max_value] range.

        Args:
            value: The float to clamp.
            min_value: The lower bound of the output range.
            max_value: The upper bound of the output range.

        Returns:
            value if within bounds, otherwise min_value or max_value.
        """
        return max(min_value, min(value, max_value))
