"""
Feature engineering module for segmentation.

Transforms raw records into a normalized feature matrix
suitable for downstream segmentation analysis.
"""

from typing import List, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler


FEATURE_KEYS = [
    "mrr",
    "growth_rate",
    "churn_rate",
    "ltv",
    "risk_score",
    "slope",
    "deviation_percentage",
]


class FeatureEngineer:
    """
    Extracts, cleans, and normalizes numeric features from a list of records.

    Does not perform clustering, labeling, or business-specific logic.
    Transformation is deterministic given the same input.
    """

    def __init__(self) -> None:
        self._scaler = StandardScaler()

    def build_feature_matrix(
        self, records: List[dict]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build a normalized feature matrix from a list of raw records.

        Args:
            records: List of dicts, each optionally containing any subset
                     of the known numeric feature keys.

        Returns:
            A tuple of:
                - feature_matrix: np.ndarray of shape (n_records, n_features),
                  standard-scaled (zero mean, unit variance).
                - feature_names: list of feature column names in matrix order.
        """
        raw = self._extract(records)
        filled = self._fill_missing(raw)
        scaled = self._scale(filled)
        return scaled, list(FEATURE_KEYS)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract(self, records: List[dict]) -> np.ndarray:
        """
        Pull the known numeric features from each record in order.

        Non-numeric values are treated as missing (replaced in next step).

        Args:
            records: Raw input records.

        Returns:
            2-D float array of shape (n_records, n_features) with np.nan
            for any missing or non-numeric field.
        """
        n = len(records)
        k = len(FEATURE_KEYS)
        matrix = np.full((n, k), np.nan, dtype=np.float64)

        for i, record in enumerate(records):
            for j, key in enumerate(FEATURE_KEYS):
                value = record.get(key)
                if value is None:
                    continue
                try:
                    matrix[i, j] = float(value)
                except (TypeError, ValueError):
                    pass  # leave as nan; handled by _fill_missing

        return matrix

    def _fill_missing(self, matrix: np.ndarray) -> np.ndarray:
        """
        Replace NaN values with 0.

        Args:
            matrix: Array possibly containing np.nan.

        Returns:
            Array with all NaN replaced by 0.0.
        """
        return np.where(np.isnan(matrix), 0.0, matrix)

    def _scale(self, matrix: np.ndarray) -> np.ndarray:
        """
        Apply standard scaling (zero mean, unit variance) column-wise.

        If a column has zero variance (constant values), StandardScaler
        will produce zeros for that column, which is safe.

        Args:
            matrix: Clean float array of shape (n_records, n_features).

        Returns:
            Scaled array of the same shape.
        """
        if matrix.shape[0] == 0:
            return matrix

        return self._scaler.fit_transform(matrix)
