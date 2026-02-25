"""
db/models/scoring.py

Normalized persistence schema for score runs and ranking snapshots.

Design goals
------------
- Dataset-scoped score storage.
- Timestamped, versioned runs for historical comparison.
- No duplication of raw metric values.
- Fast run-over-run and subject-over-time lookups.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base

if TYPE_CHECKING:
    from db.models.canonical_insight_record import CanonicalInsightRecord
    from db.models.computed_kpi import ComputedKPI
    from db.models.dataset import Dataset


class ScoringRun(Base):
    """
    One scoring execution snapshot for a single dataset.

    ``run_version`` is dataset-local and monotonic (enforced by unique key).
    """

    __tablename__ = "scoring_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
    )
    run_version: Mapped[int] = mapped_column(Integer, nullable=False)
    scoring_family: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="Logical scoring family (e.g. role_scoring, risk_priority).",
    )
    algorithm_version: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="v1",
    )
    as_of_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="Effective timestamp of source data used for this run.",
    )
    score_window_start: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    score_window_end: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    run_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Configuration, diagnostics, and confidence context for this run.",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="scoring_runs")
    relative_scores: Mapped[list["RelativeScore"]] = relationship(
        "RelativeScore",
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    composite_scores: Mapped[list["CompositeScore"]] = relationship(
        "CompositeScore",
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    ranking_results: Mapped[list["RankingResult"]] = relationship(
        "RankingResult",
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    signal_references: Mapped[list["ScoreSignalReference"]] = relationship(
        "ScoreSignalReference",
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        UniqueConstraint(
            "dataset_id",
            "run_version",
            "scoring_family",
            name="uq_scoring_runs_dataset_version_family",
        ),
        Index("ix_scoring_runs_dataset_id", "dataset_id"),
        Index("ix_scoring_runs_dataset_family", "dataset_id", "scoring_family"),
        Index("ix_scoring_runs_dataset_as_of", "dataset_id", "as_of_timestamp"),
        Index("ix_scoring_runs_created_at", "created_at"),
    )


class ScoringSubject(Base):
    """
    Stable subject dictionary for scoring (entity/dimension member).

    Keeps subject identity normalized so score tables avoid repeating labels.
    """

    __tablename__ = "scoring_subjects"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
    )
    subject_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="Dimension namespace (team, channel, region, product_line, etc.).",
    )
    subject_key: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Stable key used for historical comparisons.",
    )
    subject_label: Mapped[str | None] = mapped_column(String(255), nullable=True)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="scoring_subjects")
    relative_scores: Mapped[list["RelativeScore"]] = relationship(
        "RelativeScore",
        back_populates="subject",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    composite_scores: Mapped[list["CompositeScore"]] = relationship(
        "CompositeScore",
        back_populates="subject",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    ranking_results: Mapped[list["RankingResult"]] = relationship(
        "RankingResult",
        back_populates="subject",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    signal_references: Mapped[list["ScoreSignalReference"]] = relationship(
        "ScoreSignalReference",
        back_populates="subject",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        UniqueConstraint(
            "dataset_id",
            "subject_type",
            "subject_key",
            name="uq_scoring_subject_dataset_type_key",
        ),
        Index("ix_scoring_subject_dataset_id", "dataset_id"),
        Index("ix_scoring_subject_type_key", "subject_type", "subject_key"),
    )


class RelativeScore(Base):
    """
    Metric-level relative score for one subject in one run.
    """

    __tablename__ = "relative_scores"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("scoring_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    subject_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("scoring_subjects.id", ondelete="CASCADE"),
        nullable=False,
    )
    metric_name: Mapped[str] = mapped_column(String(120), nullable=False)
    relative_score: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    relative_method: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="zscore",
    )
    details_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Deterministic decomposition terms (without duplicating source metric values).",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    run: Mapped["ScoringRun"] = relationship("ScoringRun", back_populates="relative_scores")
    subject: Mapped["ScoringSubject"] = relationship("ScoringSubject", back_populates="relative_scores")

    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "subject_id",
            "metric_name",
            name="uq_relative_scores_run_subject_metric",
        ),
        Index("ix_relative_scores_run_metric", "run_id", "metric_name"),
        Index("ix_relative_scores_subject_metric_run", "subject_id", "metric_name", "run_id"),
        Index("ix_relative_scores_metric_value", "metric_name", "relative_score"),
    )


class CompositeScore(Base):
    """
    Composite score per subject per run.
    """

    __tablename__ = "composite_scores"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("scoring_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    subject_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("scoring_subjects.id", ondelete="CASCADE"),
        nullable=False,
    )
    composite_score: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    component_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    composite_method: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="weighted_sum",
    )
    weighting_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Weights and deterministic formula metadata.",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    run: Mapped["ScoringRun"] = relationship("ScoringRun", back_populates="composite_scores")
    subject: Mapped["ScoringSubject"] = relationship("ScoringSubject", back_populates="composite_scores")
    rankings: Mapped[list["RankingResult"]] = relationship(
        "RankingResult",
        back_populates="composite_score",
    )

    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "subject_id",
            name="uq_composite_scores_run_subject",
        ),
        Index("ix_composite_scores_run_id", "run_id"),
        Index("ix_composite_scores_run_value", "run_id", "composite_score"),
        Index("ix_composite_scores_subject_run", "subject_id", "run_id"),
    )


class RankingResult(Base):
    """
    Rank snapshot for one subject within a run and ranking dimension.
    """

    __tablename__ = "ranking_results"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("scoring_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    subject_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("scoring_subjects.id", ondelete="CASCADE"),
        nullable=False,
    )
    composite_score_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("composite_scores.id", ondelete="SET NULL"),
        nullable=True,
    )
    ranking_dimension: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="overall",
    )
    rank_position: Mapped[int] = mapped_column(Integer, nullable=False)
    percentile: Mapped[float | None] = mapped_column(Float, nullable=True)
    ranking_bucket: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    run: Mapped["ScoringRun"] = relationship("ScoringRun", back_populates="ranking_results")
    subject: Mapped["ScoringSubject"] = relationship("ScoringSubject", back_populates="ranking_results")
    composite_score: Mapped["CompositeScore | None"] = relationship(
        "CompositeScore",
        back_populates="rankings",
    )

    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "ranking_dimension",
            "subject_id",
            name="uq_ranking_results_run_dimension_subject",
        ),
        Index("ix_ranking_results_run_dimension_rank", "run_id", "ranking_dimension", "rank_position"),
        Index("ix_ranking_results_subject_dimension_run", "subject_id", "ranking_dimension", "run_id"),
    )


class ScoreSignalReference(Base):
    """
    Provenance links from scores to source records.

    Stores references only; raw metric values remain in source tables.
    """

    __tablename__ = "score_signal_references"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("scoring_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    subject_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("scoring_subjects.id", ondelete="CASCADE"),
        nullable=True,
    )
    metric_name: Mapped[str] = mapped_column(String(120), nullable=False)
    canonical_record_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("canonical_insight_records.id", ondelete="CASCADE"),
        nullable=True,
    )
    computed_kpi_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("computed_kpis.id", ondelete="CASCADE"),
        nullable=True,
    )
    source_metric_key: Mapped[str | None] = mapped_column(
        String(120),
        nullable=True,
        comment="Metric key within the referenced source row (if nested payload).",
    )
    contribution_weight: Mapped[float | None] = mapped_column(Float, nullable=True)
    reference_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    run: Mapped["ScoringRun"] = relationship("ScoringRun", back_populates="signal_references")
    subject: Mapped["ScoringSubject | None"] = relationship("ScoringSubject", back_populates="signal_references")
    canonical_record: Mapped["CanonicalInsightRecord | None"] = relationship("CanonicalInsightRecord")
    computed_kpi: Mapped["ComputedKPI | None"] = relationship("ComputedKPI")

    __table_args__ = (
        CheckConstraint(
            "(canonical_record_id IS NOT NULL) <> (computed_kpi_id IS NOT NULL)",
            name="ck_score_signal_reference_exactly_one_source",
        ),
        UniqueConstraint(
            "run_id",
            "subject_id",
            "metric_name",
            "canonical_record_id",
            "computed_kpi_id",
            "source_metric_key",
            name="uq_score_signal_reference_uniqueness",
        ),
        Index("ix_score_signal_references_run_metric", "run_id", "metric_name"),
        Index("ix_score_signal_references_canonical", "canonical_record_id"),
        Index("ix_score_signal_references_computed", "computed_kpi_id"),
    )
