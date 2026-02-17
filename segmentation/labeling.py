"""
Business labeling module for segmentation clusters.

Applies deterministic threshold-based rules to cluster profiles
to assign human-readable business labels. No ML or DB access.
"""

from typing import Any


# ------------------------------------------------------------------
# Thresholds (single source of truth — adjust here only)
# ------------------------------------------------------------------

_GROWTH_HIGH: float = 0.10       # > 10 % growth
_GROWTH_STABLE_LOW: float = 0.0  # >= 0 % (non-negative)
_GROWTH_STABLE_HIGH: float = 0.10  # <= 10 % (not high)
_GROWTH_NEGATIVE: float = 0.0    # < 0 %

_CHURN_LOW: float = 0.05         # <= 5 % churn
_CHURN_HIGH: float = 0.10        # > 10 % churn

_RISK_LOW: float = 0.30          # <= 30
_RISK_MODERATE_HIGH: float = 0.60  # <= 60
_RISK_HIGH: float = 0.60         # > 60

# ------------------------------------------------------------------
# Labels
# ------------------------------------------------------------------

_LABEL_HIGH_VALUE = "High Value / Growth Segment"
_LABEL_AT_RISK = "At Risk Segment"
_LABEL_STABLE = "Stable Segment"
_LABEL_DECLINING = "Declining Segment"
_LABEL_DEFAULT = "Unclassified Segment"


class ClusterLabeler:
    """
    Assigns a business label to each cluster based on its profile metrics.

    Responsibilities:
        - Evaluate deterministic threshold rules in priority order.
        - Annotate each cluster profile with a ``business_label`` key.

    Not responsible for:
        - Computing profile metrics (that is the profiler's job).
        - Any ML inference or model calls.
        - DB access or persistence.
    """

    def label(self, profile: dict) -> dict:
        """
        Annotate each cluster in a profile dict with a business label.

        Args:
            profile: Dict produced by ``ClusterProfiler.profile_clusters``,
                     mapping cluster_id -> metric dict containing at minimum:
                     ``avg_growth``, ``avg_churn``, ``avg_risk``, ``avg_ltv``.

        Returns:
            A new dict with the same structure, each cluster extended with::

                { "business_label": str }

            The original profile dict is not mutated.
        """
        return {
            cluster_id: self._annotate(metrics)
            for cluster_id, metrics in profile.items()
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _annotate(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """
        Copy a single cluster's metrics and append a business label.

        Args:
            metrics: Metric dict for one cluster.

        Returns:
            New dict containing all original keys plus ``business_label``.
        """
        growth = float(metrics.get("avg_growth", 0.0))
        churn = float(metrics.get("avg_churn", 0.0))
        risk = float(metrics.get("avg_risk", 0.0))

        return {
            **metrics,
            "business_label": self._resolve_label(growth, churn, risk),
        }

    @staticmethod
    def _resolve_label(growth: float, churn: float, risk: float) -> str:
        """
        Apply threshold rules in priority order and return the matching label.

        Priority:
            1. High Value / Growth Segment
            2. At Risk Segment
            3. Declining Segment
            4. Stable Segment
            5. Unclassified Segment (fallback)

        Args:
            growth: Average growth rate for the cluster.
            churn:  Average churn rate for the cluster.
            risk:   Average risk score for the cluster.

        Returns:
            Business label string.
        """
        if growth > _GROWTH_HIGH and churn <= _CHURN_LOW and risk <= _RISK_LOW:
            return _LABEL_HIGH_VALUE

        if growth <= _GROWTH_STABLE_LOW and churn > _CHURN_HIGH:
            return _LABEL_AT_RISK

        if growth < _GROWTH_NEGATIVE and risk > _RISK_HIGH:
            return _LABEL_DECLINING

        if (
            _GROWTH_STABLE_LOW <= growth <= _GROWTH_STABLE_HIGH
            and risk <= _RISK_MODERATE_HIGH
        ):
            return _LABEL_STABLE

        return _LABEL_DEFAULT
