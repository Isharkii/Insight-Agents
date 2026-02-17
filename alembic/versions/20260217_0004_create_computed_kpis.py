"""create computed_kpis table

Revision ID: 20260217_0004
Revises: 20260217_0003
Create Date: 2026-02-17 18:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260217_0004"
down_revision = "20260217_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "computed_kpis",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("entity_name", sa.String(length=255), nullable=False,
                  comment="Client or competitor entity name"),
        sa.Column("period_start", sa.DateTime(timezone=True), nullable=False,
                  comment="Inclusive start of the measurement period (UTC)"),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=False,
                  comment="Inclusive end of the measurement period (UTC)"),
        sa.Column(
            "computed_kpis",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            comment="Structured KPI results keyed by metric name",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "entity_name",
            "period_start",
            "period_end",
            name="uq_computed_kpis_entity_period",
        ),
    )
    op.create_index(
        "ix_computed_kpis_entity_name",
        "computed_kpis",
        ["entity_name"],
        unique=False,
    )
    op.create_index(
        "ix_computed_kpis_period_start",
        "computed_kpis",
        ["period_start"],
        unique=False,
    )
    op.create_index(
        "ix_computed_kpis_entity_period_start",
        "computed_kpis",
        ["entity_name", "period_start"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_computed_kpis_entity_period_start", table_name="computed_kpis")
    op.drop_index("ix_computed_kpis_period_start", table_name="computed_kpis")
    op.drop_index("ix_computed_kpis_entity_name", table_name="computed_kpis")
    op.drop_table("computed_kpis")
