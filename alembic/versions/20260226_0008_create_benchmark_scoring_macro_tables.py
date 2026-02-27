"""Create benchmark, scoring, and macro metric tables.

Revision ID: 20260226_0008
Revises: 20260224_0007
Create Date: 2026-02-26

Adds the following 12 tables:
  - industry_categories
  - benchmarks
  - benchmark_metrics
  - benchmark_snapshots
  - macro_metric_runs
  - macro_metrics
  - scoring_runs
  - scoring_subjects
  - relative_scores
  - composite_scores
  - ranking_results
  - score_signal_references
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260226_0008"
down_revision = "20260224_0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ---- industry_categories ----
    op.create_table(
        "industry_categories",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("industry_key", sa.String(64), nullable=False),
        sa.Column("industry_name", sa.String(255), nullable=False),
        sa.Column("metadata_json", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("industry_key", name="uq_industry_categories_industry_key"),
    )
    op.create_index("ix_industry_categories_industry_key", "industry_categories", ["industry_key"])
    op.create_index("ix_industry_categories_industry_name", "industry_categories", ["industry_name"])

    # ---- benchmarks ----
    op.create_table(
        "benchmarks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("benchmark_name", sa.String(255), nullable=False),
        sa.Column("industry", sa.String(120), nullable=False),
        sa.Column("industry_category_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("source", sa.String(64), nullable=True),
        sa.Column("metadata_json", postgresql.JSONB(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["industry_category_id"], ["industry_categories.id"], ondelete="SET NULL"),
        sa.UniqueConstraint("benchmark_name", "industry", name="uq_benchmarks_name_industry"),
    )
    op.create_index("ix_benchmarks_industry", "benchmarks", ["industry"])
    op.create_index("ix_benchmarks_benchmark_name", "benchmarks", ["benchmark_name"])
    op.create_index("ix_benchmarks_industry_category_id", "benchmarks", ["industry_category_id"])
    op.create_index("ix_benchmarks_is_active", "benchmarks", ["is_active"])

    # ---- benchmark_metrics ----
    op.create_table(
        "benchmark_metrics",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("benchmark_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("metric_name", sa.String(120), nullable=False),
        sa.Column("unit", sa.String(32), nullable=True),
        sa.Column("metric_config_json", postgresql.JSONB(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["benchmark_id"], ["benchmarks.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("benchmark_id", "metric_name", name="uq_benchmark_metrics_benchmark_metric_name"),
    )
    op.create_index("ix_benchmark_metrics_metric_name", "benchmark_metrics", ["metric_name"])
    op.create_index("ix_benchmark_metrics_benchmark_id", "benchmark_metrics", ["benchmark_id"])
    op.create_index("ix_benchmark_metrics_is_active", "benchmark_metrics", ["is_active"])

    # ---- benchmark_snapshots ----
    op.create_table(
        "benchmark_snapshots",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("benchmark_metric_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("period_start", sa.Date(), nullable=False),
        sa.Column("period_end", sa.Date(), nullable=False),
        sa.Column("frequency", sa.String(1), nullable=False),
        sa.Column("snapshot_version", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("metric_value", sa.Numeric(20, 6), nullable=True),
        sa.Column("metric_payload_json", postgresql.JSONB(), nullable=True),
        sa.Column("metadata_json", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["benchmark_metric_id"], ["benchmark_metrics.id"], ondelete="CASCADE"),
        sa.CheckConstraint("frequency IN ('M', 'Q', 'Y')", name="ck_benchmark_snapshots_frequency"),
        sa.CheckConstraint(
            "metric_value IS NOT NULL OR metric_payload_json IS NOT NULL",
            name="ck_benchmark_snapshots_value_or_payload",
        ),
        sa.UniqueConstraint(
            "benchmark_metric_id", "period_end", "snapshot_version",
            name="uq_benchmark_snapshots_metric_period_version",
        ),
    )
    op.create_index("ix_benchmark_snapshots_period_end", "benchmark_snapshots", ["period_end"])
    op.create_index("ix_benchmark_snapshots_metric_period_end", "benchmark_snapshots", ["benchmark_metric_id", "period_end"])
    op.create_index("ix_benchmark_snapshots_period_frequency", "benchmark_snapshots", ["period_end", "frequency"])

    # ---- macro_metric_runs ----
    op.create_table(
        "macro_metric_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_key", sa.String(32), nullable=False),
        sa.Column("country_code", sa.String(3), nullable=False),
        sa.Column("run_version", sa.Integer(), nullable=False),
        sa.Column("source_release_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("is_current", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("metadata_json", postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "source_key", "country_code", "run_version",
            name="uq_macro_metric_runs_source_country_version",
        ),
    )
    op.create_index("ix_macro_metric_runs_country_code", "macro_metric_runs", ["country_code"])
    op.create_index("ix_macro_metric_runs_source_key", "macro_metric_runs", ["source_key"])
    op.create_index(
        "ix_macro_metric_runs_source_country_current", "macro_metric_runs",
        ["source_key", "country_code", "is_current"],
    )

    # ---- macro_metrics ----
    op.create_table(
        "macro_metrics",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("country_code", sa.String(3), nullable=False),
        sa.Column("metric_name", sa.String(64), nullable=False),
        sa.Column("frequency", sa.String(1), nullable=False),
        sa.Column("period_start", sa.Date(), nullable=False),
        sa.Column("period_end", sa.Date(), nullable=False),
        sa.Column("value", sa.Numeric(20, 6), nullable=False),
        sa.Column("unit", sa.String(32), nullable=True),
        sa.Column("metadata_json", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["run_id"], ["macro_metric_runs.id"], ondelete="CASCADE"),
        sa.CheckConstraint("frequency IN ('M', 'Q')", name="ck_macro_metrics_frequency"),
        sa.UniqueConstraint(
            "run_id", "country_code", "metric_name", "frequency", "period_end",
            name="uq_macro_metrics_run_country_metric_frequency_period_end",
        ),
    )
    op.create_index("ix_macro_metrics_country_code", "macro_metrics", ["country_code"])
    op.create_index("ix_macro_metrics_metric_name", "macro_metrics", ["metric_name"])
    op.create_index("ix_macro_metrics_period_end", "macro_metrics", ["period_end"])
    op.create_index(
        "ix_macro_metrics_country_metric_frequency_period_end", "macro_metrics",
        ["country_code", "metric_name", "frequency", "period_end"],
    )
    op.create_index("ix_macro_metrics_run_id", "macro_metrics", ["run_id"])

    # ---- scoring_runs ----
    op.create_table(
        "scoring_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("dataset_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_version", sa.Integer(), nullable=False),
        sa.Column("scoring_family", sa.String(64), nullable=False),
        sa.Column("algorithm_version", sa.String(64), nullable=False, server_default=sa.text("'v1'")),
        sa.Column("as_of_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("score_window_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("score_window_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("run_metadata", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.UniqueConstraint(
            "dataset_id", "run_version", "scoring_family",
            name="uq_scoring_runs_dataset_version_family",
        ),
    )
    op.create_index("ix_scoring_runs_dataset_id", "scoring_runs", ["dataset_id"])
    op.create_index("ix_scoring_runs_dataset_family", "scoring_runs", ["dataset_id", "scoring_family"])
    op.create_index("ix_scoring_runs_dataset_as_of", "scoring_runs", ["dataset_id", "as_of_timestamp"])
    op.create_index("ix_scoring_runs_created_at", "scoring_runs", ["created_at"])

    # ---- scoring_subjects ----
    op.create_table(
        "scoring_subjects",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("dataset_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("subject_type", sa.String(64), nullable=False),
        sa.Column("subject_key", sa.String(255), nullable=False),
        sa.Column("subject_label", sa.String(255), nullable=True),
        sa.Column("metadata_json", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.UniqueConstraint(
            "dataset_id", "subject_type", "subject_key",
            name="uq_scoring_subject_dataset_type_key",
        ),
    )
    op.create_index("ix_scoring_subject_dataset_id", "scoring_subjects", ["dataset_id"])
    op.create_index("ix_scoring_subject_type_key", "scoring_subjects", ["subject_type", "subject_key"])

    # ---- relative_scores ----
    op.create_table(
        "relative_scores",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("subject_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("metric_name", sa.String(120), nullable=False),
        sa.Column("relative_score", sa.Float(), nullable=False),
        sa.Column("confidence_score", sa.Float(), nullable=True),
        sa.Column("relative_method", sa.String(64), nullable=False, server_default=sa.text("'zscore'")),
        sa.Column("details_json", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["run_id"], ["scoring_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["subject_id"], ["scoring_subjects.id"], ondelete="CASCADE"),
        sa.UniqueConstraint(
            "run_id", "subject_id", "metric_name",
            name="uq_relative_scores_run_subject_metric",
        ),
    )
    op.create_index("ix_relative_scores_run_metric", "relative_scores", ["run_id", "metric_name"])
    op.create_index("ix_relative_scores_subject_metric_run", "relative_scores", ["subject_id", "metric_name", "run_id"])
    op.create_index("ix_relative_scores_metric_value", "relative_scores", ["metric_name", "relative_score"])

    # ---- composite_scores ----
    op.create_table(
        "composite_scores",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("subject_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("composite_score", sa.Float(), nullable=False),
        sa.Column("confidence_score", sa.Float(), nullable=True),
        sa.Column("component_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("composite_method", sa.String(64), nullable=False, server_default=sa.text("'weighted_sum'")),
        sa.Column("weighting_json", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["run_id"], ["scoring_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["subject_id"], ["scoring_subjects.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("run_id", "subject_id", name="uq_composite_scores_run_subject"),
    )
    op.create_index("ix_composite_scores_run_id", "composite_scores", ["run_id"])
    op.create_index("ix_composite_scores_run_value", "composite_scores", ["run_id", "composite_score"])
    op.create_index("ix_composite_scores_subject_run", "composite_scores", ["subject_id", "run_id"])

    # ---- ranking_results ----
    op.create_table(
        "ranking_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("subject_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("composite_score_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("ranking_dimension", sa.String(64), nullable=False, server_default=sa.text("'overall'")),
        sa.Column("rank_position", sa.Integer(), nullable=False),
        sa.Column("percentile", sa.Float(), nullable=True),
        sa.Column("ranking_bucket", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["run_id"], ["scoring_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["subject_id"], ["scoring_subjects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["composite_score_id"], ["composite_scores.id"], ondelete="SET NULL"),
        sa.UniqueConstraint(
            "run_id", "ranking_dimension", "subject_id",
            name="uq_ranking_results_run_dimension_subject",
        ),
    )
    op.create_index("ix_ranking_results_run_dimension_rank", "ranking_results", ["run_id", "ranking_dimension", "rank_position"])
    op.create_index("ix_ranking_results_subject_dimension_run", "ranking_results", ["subject_id", "ranking_dimension", "run_id"])

    # ---- score_signal_references ----
    op.create_table(
        "score_signal_references",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("subject_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("metric_name", sa.String(120), nullable=False),
        sa.Column("canonical_record_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("computed_kpi_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("source_metric_key", sa.String(120), nullable=True),
        sa.Column("contribution_weight", sa.Float(), nullable=True),
        sa.Column("reference_metadata", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["run_id"], ["scoring_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["subject_id"], ["scoring_subjects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["canonical_record_id"], ["canonical_insight_records.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["computed_kpi_id"], ["computed_kpis.id"], ondelete="CASCADE"),
        sa.CheckConstraint(
            "(canonical_record_id IS NOT NULL) <> (computed_kpi_id IS NOT NULL)",
            name="ck_score_signal_reference_exactly_one_source",
        ),
        sa.UniqueConstraint(
            "run_id", "subject_id", "metric_name", "canonical_record_id", "computed_kpi_id", "source_metric_key",
            name="uq_score_signal_reference_uniqueness",
        ),
    )
    op.create_index("ix_score_signal_references_run_metric", "score_signal_references", ["run_id", "metric_name"])
    op.create_index("ix_score_signal_references_canonical", "score_signal_references", ["canonical_record_id"])
    op.create_index("ix_score_signal_references_computed", "score_signal_references", ["computed_kpi_id"])


def downgrade() -> None:
    op.drop_table("score_signal_references")
    op.drop_table("ranking_results")
    op.drop_table("composite_scores")
    op.drop_table("relative_scores")
    op.drop_table("scoring_subjects")
    op.drop_table("scoring_runs")
    op.drop_table("macro_metrics")
    op.drop_table("macro_metric_runs")
    op.drop_table("benchmark_snapshots")
    op.drop_table("benchmark_metrics")
    op.drop_table("benchmarks")
    op.drop_table("industry_categories")
