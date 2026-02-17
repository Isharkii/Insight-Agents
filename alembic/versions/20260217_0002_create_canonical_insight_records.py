"""create canonical_insight_records table

Revision ID: 20260217_0002
Revises: 20260217_0001
Create Date: 2026-02-17 16:40:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20260217_0002"
down_revision = "20260217_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "canonical_insight_records",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_type", sa.String(length=32), nullable=False),
        sa.Column("entity_name", sa.String(length=255), nullable=False),
        sa.Column("category", sa.String(length=32), nullable=False),
        sa.Column("metric_name", sa.String(length=120), nullable=False),
        sa.Column("metric_value", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("region", sa.String(length=120), nullable=True),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "source_type",
            "entity_name",
            "category",
            "metric_name",
            "timestamp",
            name="uq_canonical_insight_records_dedupe",
        ),
    )
    op.create_index(
        "ix_canonical_insight_records_source_type",
        "canonical_insight_records",
        ["source_type"],
        unique=False,
    )
    op.create_index(
        "ix_canonical_insight_records_entity_name",
        "canonical_insight_records",
        ["entity_name"],
        unique=False,
    )
    op.create_index(
        "ix_canonical_insight_records_category",
        "canonical_insight_records",
        ["category"],
        unique=False,
    )
    op.create_index(
        "ix_canonical_insight_records_metric_name",
        "canonical_insight_records",
        ["metric_name"],
        unique=False,
    )
    op.create_index(
        "ix_canonical_insight_records_timestamp",
        "canonical_insight_records",
        ["timestamp"],
        unique=False,
    )
    op.create_index(
        "ix_canonical_insight_records_region",
        "canonical_insight_records",
        ["region"],
        unique=False,
    )
    op.create_index(
        "ix_canonical_insight_records_entity_category_timestamp",
        "canonical_insight_records",
        ["entity_name", "category", "timestamp"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_canonical_insight_records_entity_category_timestamp",
        table_name="canonical_insight_records",
    )
    op.drop_index("ix_canonical_insight_records_region", table_name="canonical_insight_records")
    op.drop_index("ix_canonical_insight_records_timestamp", table_name="canonical_insight_records")
    op.drop_index("ix_canonical_insight_records_metric_name", table_name="canonical_insight_records")
    op.drop_index("ix_canonical_insight_records_category", table_name="canonical_insight_records")
    op.drop_index("ix_canonical_insight_records_entity_name", table_name="canonical_insight_records")
    op.drop_index("ix_canonical_insight_records_source_type", table_name="canonical_insight_records")
    op.drop_table("canonical_insight_records")
