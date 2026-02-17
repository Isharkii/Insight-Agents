"""
Segmentation orchestrator.

Wires together feature engineering, clustering, profiling, labeling,
and persistence into a single deterministic pipeline call.
No math, no business rules, and no DB logic beyond the repository call.
"""

from datetime import date
from typing import List, Optional

from sqlalchemy.orm import Session

from segmentation.clustering import KMeansSegmentation
from segmentation.features import FeatureEngineer
from segmentation.labeling import ClusterLabeler
from segmentation.profiling import ClusterProfiler
from segmentation.repository import SegmentInsightRepository


class SegmentationOrchestrator:
    """
    Coordinates the end-to-end segmentation pipeline.

    Each pipeline step is delegated entirely to its dedicated module:
        1. FeatureEngineer   — builds the normalized feature matrix.
        2. KMeansSegmentation — assigns cluster labels and centroids.
        3. ClusterProfiler   — computes per-cluster aggregate metrics.
        4. ClusterLabeler    — attaches human-readable business labels.
        5. SegmentInsightRepository — persists the final result.

    The orchestrator itself contains no clustering math, feature logic,
    profiling logic, or business rules.

    Args:
        session: Optional SQLAlchemy session for persistence.
                 When ``None``, the pipeline runs fully in-memory and
                 the repository step is skipped (useful for testing or
                 local dry-runs).
    """

    def __init__(self, session: Optional[Session] = None) -> None:
        self._session = session
        self._feature_engineer = FeatureEngineer()
        self._clusterer = KMeansSegmentation()
        self._profiler = ClusterProfiler()
        self._labeler = ClusterLabeler()
        self._repository = SegmentInsightRepository()

    def run_segmentation(
        self,
        entity_name: str,
        records: List[dict],
        n_clusters: int,
    ) -> dict:
        """
        Execute the full segmentation pipeline for one entity.

        Args:
            entity_name: Identifier for the entity being segmented
                         (e.g. account name or client ID).
            records:     List of raw metric dicts, one per observation.
            n_clusters:  Number of clusters (k) for KMeans.

        Returns:
            Pipeline result dict::

                {
                    "entity_name": str,
                    "n_clusters":  int,
                    "segments":    dict   # labeled cluster profiles
                }

        Raises:
            ValueError: Propagated from clustering if inputs are invalid
                        (e.g. n_clusters > n_samples).
        """
        # Step 1 — Feature engineering
        feature_matrix, _feature_names = self._feature_engineer.build_feature_matrix(
            records
        )

        # Step 2 — KMeans clustering
        labels, _centroids = self._clusterer.cluster(
            features=feature_matrix,
            n_clusters=n_clusters,
        )

        # Step 3 — Cluster profiling
        profiles = self._profiler.profile_clusters(
            records=records,
            labels=labels,
        )

        # Step 4 — Business labeling
        labeled_segments = self._labeler.label(profiles)

        # Step 5 — Persistence (skipped when no session is provided)
        self._persist(
            entity_name=entity_name,
            n_clusters=n_clusters,
            segment_data=labeled_segments,
        )

        return {
            "entity_name": entity_name,
            "n_clusters": n_clusters,
            "segments": labeled_segments,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _persist(
        self,
        entity_name: str,
        n_clusters: int,
        segment_data: dict,
    ) -> None:
        """
        Save the segmentation result via the repository if a session exists.

        Uses today's date as ``period_end`` to mark the analysis snapshot.
        The caller is responsible for committing or rolling back the session.

        Args:
            entity_name:  Entity identifier.
            n_clusters:   Number of clusters used.
            segment_data: Labeled cluster profiles to persist.
        """
        if self._session is None:
            return

        period_end = date.today().isoformat()   # "YYYY-MM-DD"

        self._repository.save_segments(
            session=self._session,
            entity_name=entity_name,
            period_end=period_end,
            n_clusters=n_clusters,
            segment_data=segment_data,
        )
