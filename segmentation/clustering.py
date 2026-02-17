"""
KMeans clustering engine for segmentation.

Accepts a pre-built feature matrix and returns cluster assignments
and centroids. No feature engineering or business logic here.
"""

from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans


_RANDOM_STATE = 42
_N_INIT = 10


class KMeansSegmentation:
    """
    Thin, deterministic wrapper around sklearn KMeans.

    Responsibilities:
        - Fit KMeans on a provided feature matrix.
        - Return per-record cluster labels and cluster centroids.

    Not responsible for:
        - Feature engineering or scaling.
        - Choosing optimal k.
        - Profiling or labeling clusters.
        - Any persistence or DB access.
    """

    def __init__(self) -> None:
        self._model: KMeans | None = None

    def cluster(
        self,
        features: np.ndarray,
        n_clusters: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit KMeans and return labels and centroids.

        Args:
            features:   2-D float array of shape (n_samples, n_features).
                        Must be pre-scaled; no transformation is applied here.
            n_clusters: Number of clusters (k). Must satisfy
                        1 <= n_clusters <= n_samples.

        Returns:
            A tuple of:
                - labels:     1-D int array of shape (n_samples,) with the
                              cluster index assigned to each record.
                - centroids:  2-D float array of shape (n_clusters, n_features)
                              with the centroid of each cluster.

        Raises:
            ValueError: If the feature array is empty or n_clusters is invalid.
        """
        self._validate(features, n_clusters)

        self._model = KMeans(
            n_clusters=n_clusters,
            random_state=_RANDOM_STATE,
            n_init=_N_INIT,
        )
        self._model.fit(features)

        labels: np.ndarray = self._model.labels_.astype(np.int32)
        centroids: np.ndarray = self._model.cluster_centers_.astype(np.float64)

        return labels, centroids

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(features: np.ndarray, n_clusters: int) -> None:
        """
        Sanity-check inputs before fitting.

        Args:
            features:   Feature matrix to validate.
            n_clusters: Requested cluster count.

        Raises:
            ValueError: On any invalid condition.
        """
        if features.ndim != 2:
            raise ValueError(
                f"features must be a 2-D array, got shape {features.shape}."
            )

        n_samples = features.shape[0]

        if n_samples == 0:
            raise ValueError("features array is empty (0 samples).")

        if not isinstance(n_clusters, int) or n_clusters < 1:
            raise ValueError(
                f"n_clusters must be a positive integer, got {n_clusters!r}."
            )

        if n_clusters > n_samples:
            raise ValueError(
                f"n_clusters ({n_clusters}) cannot exceed "
                f"number of samples ({n_samples})."
            )
