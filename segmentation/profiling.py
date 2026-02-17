"""
Cluster profiling module for segmentation.

Computes per-cluster aggregate statistics from raw records
and their assigned cluster labels. No ML or feature engineering here.
"""

from collections import defaultdict
from typing import List

import numpy as np


_PROFILE_KEYS = ("growth_rate", "churn_rate", "risk_score", "ltv")


class ClusterProfiler:
    """
    Summarises clusters by computing mean values of key business metrics.

    Responsibilities:
        - Group records by cluster label.
        - Compute per-cluster averages for a fixed set of numeric fields.
        - Return a plain dict keyed by cluster id.

    Not responsible for:
        - Feature engineering or scaling.
        - Cluster assignment (labels come from the clustering step).
        - Any DB access or persistence.
        - ML inference.
    """

    def profile_clusters(
        self,
        records: List[dict],
        labels: np.ndarray,
    ) -> dict:
        """
        Build a profile summary for each cluster.

        Args:
            records: Raw input records aligned with ``labels``.
                     Each dict may contain any subset of the profile keys;
                     missing or non-numeric values are treated as 0.
            labels:  1-D integer array of cluster assignments,
                     same length as ``records``.

        Returns:
            Dict mapping cluster_id (int) to a profile dict::

                {
                    cluster_id: {
                        "size": int,
                        "avg_growth": float,
                        "avg_churn": float,
                        "avg_risk": float,
                        "avg_ltv": float,
                    },
                    ...
                }

        Raises:
            ValueError: If ``records`` and ``labels`` differ in length.
        """
        self._validate(records, labels)

        buckets: dict[int, dict[str, list]] = defaultdict(
            lambda: {key: [] for key in _PROFILE_KEYS}
        )

        for record, label in zip(records, labels):
            cluster_id = int(label)
            for key in _PROFILE_KEYS:
                buckets[cluster_id][key].append(self._to_float(record.get(key)))

        return {
            cluster_id: self._summarise(values)
            for cluster_id, values in sorted(buckets.items())
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_float(value) -> float:
        """
        Coerce a value to float; return 0.0 for missing or non-numeric input.

        Args:
            value: Raw field value from a record.

        Returns:
            Numeric float, defaulting to 0.0 on failure.
        """
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _mean(values: list) -> float:
        """
        Compute the arithmetic mean of a list of floats.

        Returns 0.0 for an empty list.

        Args:
            values: List of numeric values.

        Returns:
            Mean as a Python float.
        """
        if not values:
            return 0.0
        return float(np.mean(values))

    def _summarise(self, values: dict[str, list]) -> dict:
        """
        Convert per-key value lists into the final profile structure.

        Args:
            values: Mapping of profile key -> list of floats for one cluster.

        Returns:
            Profile dict with ``size`` and the four ``avg_*`` fields.
        """
        size = len(values[_PROFILE_KEYS[0]])
        return {
            "size": size,
            "avg_growth": self._mean(values["growth_rate"]),
            "avg_churn": self._mean(values["churn_rate"]),
            "avg_risk": self._mean(values["risk_score"]),
            "avg_ltv": self._mean(values["ltv"]),
        }

    @staticmethod
    def _validate(records: List[dict], labels: np.ndarray) -> None:
        """
        Ensure records and labels are compatible.

        Args:
            records: Input record list.
            labels:  Cluster label array.

        Raises:
            ValueError: If lengths differ.
        """
        if len(records) != len(labels):
            raise ValueError(
                f"records and labels must have the same length; "
                f"got {len(records)} records and {len(labels)} labels."
            )
