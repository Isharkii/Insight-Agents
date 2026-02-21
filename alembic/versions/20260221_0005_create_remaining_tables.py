"""create clients, datasets, analytical_metrics, insights, forecast_metric, business_risk_scores

Revision ID: 20260221_0005
Revises: 20260217_0004
Create Date: 2026-02-21 00:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260221_0005"
down_revision = "20260217_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ---------------------------------------------------------------------------
    # clients
    # No foreign keys. Created first; all other tables reference it.
    # ---------------------------------------------------------------------------
    op.create_table(
        "clients",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("domain", sa.String(length=120), nullable=True),
        sa.Column(
            "config",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Optional client-level configuration",
        ),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("name", name="uq_clients_name"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_clients_name", "clients", ["name"])
    op.create_index("ix_clients_domain", "clients", ["domain"])
    op.create_index("ix_clients_is_active", "clients", ["is_active"])

    # ---------------------------------------------------------------------------
    # datasets
    # FK → clients.id ON DELETE CASCADE
    # ---------------------------------------------------------------------------
    op.create_table(
        "datasets",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("client_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column(
            "source_type",
            sa.String(length=50),
            nullable=False,
            comment="csv, excel, api, manual",
        ),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("file_name", sa.String(length=255), nullable=True),
        sa.Column("file_path", sa.String(length=500), nullable=True),
        sa.Column("mime_type", sa.String(length=120), nullable=True),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("checksum", sa.String(length=128), nullable=True),
        sa.Column("row_count", sa.Integer(), nullable=True),
        sa.Column(
            "schema_meta",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "file_meta",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Additional uploaded-file metadata",
        ),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["client_id"],
            ["clients.id"],
            name="fk_datasets_client_id_clients",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_datasets_client_id", "datasets", ["client_id"])
    op.create_index("ix_datasets_status", "datasets", ["status"])
    op.create_index("ix_datasets_source_type", "datasets", ["source_type"])
    op.create_index("ix_datasets_processed_at", "datasets", ["processed_at"])
    op.create_index("ix_datasets_client_status", "datasets", ["client_id", "status"])

    # ---------------------------------------------------------------------------
    # analytical_metrics
    # FK → clients.id ON DELETE CASCADE
    # FK → datasets.id ON DELETE SET NULL
    # ---------------------------------------------------------------------------
    op.create_table(
        "analytical_metrics",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("client_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("dataset_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "metric_type",
            sa.String(length=32),
            nullable=False,
            comment="kpi, trend, anomaly",
        ),
        sa.Column(
            "metric_key",
            sa.String(length=120),
            nullable=False,
            comment="Stable metric identifier",
        ),
        sa.Column("metric_label", sa.String(length=255), nullable=True),
        sa.Column("metric_value", sa.Numeric(precision=20, scale=6), nullable=True),
        sa.Column("unit", sa.String(length=40), nullable=True),
        sa.Column("period_start", sa.Date(), nullable=True),
        sa.Column("period_end", sa.Date(), nullable=True),
        sa.Column(
            "measured_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "dimensions",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Dynamic grouping dimensions (region, channel, product)",
        ),
        sa.Column(
            "metric_payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Flexible structured metric data",
        ),
        sa.Column("source", sa.String(length=80), nullable=True),
        sa.Column(
            "confidence_score",
            sa.Numeric(precision=5, scale=4),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["client_id"],
            ["clients.id"],
            name="fk_analytical_metrics_client_id_clients",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["datasets.id"],
            name="fk_analytical_metrics_dataset_id_datasets",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_analytical_metrics_client_metric_measured",
        "analytical_metrics",
        ["client_id", "metric_key", "measured_at"],
    )
    op.create_index(
        "ix_analytical_metrics_client_type_measured",
        "analytical_metrics",
        ["client_id", "metric_type", "measured_at"],
    )
    op.create_index(
        "ix_analytical_metrics_dataset_measured",
        "analytical_metrics",
        ["dataset_id", "measured_at"],
    )
    op.create_index(
        "ix_analytical_metrics_client_metric_period",
        "analytical_metrics",
        ["client_id", "metric_key", "period_start"],
    )

    # ---------------------------------------------------------------------------
    # insights
    # FK → clients.id ON DELETE CASCADE
    # FK → datasets.id ON DELETE SET NULL
    # FK → analytical_metrics.id ON DELETE SET NULL
    # ---------------------------------------------------------------------------
    op.create_table(
        "insights",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("client_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("dataset_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("metric_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("insight_type", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column(
            "output_payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            comment="Structured JSON output from the reasoning layer",
        ),
        sa.Column(
            "evidence_payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("priority", sa.String(length=20), nullable=True),
        sa.Column(
            "confidence_score",
            sa.Numeric(precision=5, scale=4),
            nullable=True,
        ),
        sa.Column(
            "generated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["client_id"],
            ["clients.id"],
            name="fk_insights_client_id_clients",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["datasets.id"],
            name="fk_insights_dataset_id_datasets",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["metric_id"],
            ["analytical_metrics.id"],
            name="fk_insights_metric_id_analytical_metrics",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_insights_client_generated_at",
        "insights",
        ["client_id", "generated_at"],
    )
    op.create_index(
        "ix_insights_client_type_generated_at",
        "insights",
        ["client_id", "insight_type", "generated_at"],
    )
    op.create_index(
        "ix_insights_dataset_generated_at",
        "insights",
        ["dataset_id", "generated_at"],
    )
    op.create_index(
        "ix_insights_metric_generated_at",
        "insights",
        ["metric_id", "generated_at"],
    )

    # ---------------------------------------------------------------------------
    # forecast_metric
    # No foreign keys. Self-contained entity-scoped table.
    # created_at has no server_default (Python-side default in ORM).
    # ---------------------------------------------------------------------------
    op.create_table(
        "forecast_metric",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("entity_name", sa.String(length=255), nullable=False),
        sa.Column("metric_name", sa.String(length=255), nullable=False),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "forecast_data",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    # Individual column indexes (index=True on mapped_column)
    op.create_index("ix_forecast_metric_entity_name", "forecast_metric", ["entity_name"])
    op.create_index("ix_forecast_metric_metric_name", "forecast_metric", ["metric_name"])
    op.create_index("ix_forecast_metric_period_end", "forecast_metric", ["period_end"])
    # Composite index from __table_args__
    op.create_index(
        "ix_forecast_metric_entity_metric_period",
        "forecast_metric",
        ["entity_name", "metric_name", "period_end"],
    )

    # ---------------------------------------------------------------------------
    # business_risk_scores
    # No foreign keys. Self-contained entity-scoped table.
    # ---------------------------------------------------------------------------
    op.create_table(
        "business_risk_scores",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("entity_name", sa.String(length=255), nullable=False),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("risk_score", sa.Integer(), nullable=False),
        sa.Column(
            "risk_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    # Individual column indexes (index=True on mapped_column)
    op.create_index(
        "ix_business_risk_scores_entity_name",
        "business_risk_scores",
        ["entity_name"],
    )
    op.create_index(
        "ix_business_risk_scores_period_end",
        "business_risk_scores",
        ["period_end"],
    )
    # Composite index from __table_args__
    op.create_index(
        "ix_brs_entity_period",
        "business_risk_scores",
        ["entity_name", "period_end"],
    )


def downgrade() -> None:
    # Drop in strict reverse dependency order.
    # Indexes are dropped implicitly with their table; explicit drops are
    # included here for clarity and to mirror upgrade() exactly.

    op.drop_index("ix_brs_entity_period", table_name="business_risk_scores")
    op.drop_index("ix_business_risk_scores_period_end", table_name="business_risk_scores")
    op.drop_index("ix_business_risk_scores_entity_name", table_name="business_risk_scores")
    op.drop_table("business_risk_scores")

    op.drop_index("ix_forecast_metric_entity_metric_period", table_name="forecast_metric")
    op.drop_index("ix_forecast_metric_period_end", table_name="forecast_metric")
    op.drop_index("ix_forecast_metric_metric_name", table_name="forecast_metric")
    op.drop_index("ix_forecast_metric_entity_name", table_name="forecast_metric")
    op.drop_table("forecast_metric")

    op.drop_index("ix_insights_metric_generated_at", table_name="insights")
    op.drop_index("ix_insights_dataset_generated_at", table_name="insights")
    op.drop_index("ix_insights_client_type_generated_at", table_name="insights")
    op.drop_index("ix_insights_client_generated_at", table_name="insights")
    op.drop_table("insights")

    op.drop_index("ix_analytical_metrics_client_metric_period", table_name="analytical_metrics")
    op.drop_index("ix_analytical_metrics_dataset_measured", table_name="analytical_metrics")
    op.drop_index("ix_analytical_metrics_client_type_measured", table_name="analytical_metrics")
    op.drop_index("ix_analytical_metrics_client_metric_measured", table_name="analytical_metrics")
    op.drop_table("analytical_metrics")

    op.drop_index("ix_datasets_client_status", table_name="datasets")
    op.drop_index("ix_datasets_processed_at", table_name="datasets")
    op.drop_index("ix_datasets_source_type", table_name="datasets")
    op.drop_index("ix_datasets_status", table_name="datasets")
    op.drop_index("ix_datasets_client_id", table_name="datasets")
    op.drop_table("datasets")

    op.drop_index("ix_clients_is_active", table_name="clients")
    op.drop_index("ix_clients_domain", table_name="clients")
    op.drop_index("ix_clients_name", table_name="clients")
    op.drop_table("clients")
