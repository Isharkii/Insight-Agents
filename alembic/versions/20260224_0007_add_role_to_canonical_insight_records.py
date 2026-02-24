"""add role column to canonical_insight_records

Revision ID: 20260224_0007
Revises: 20260221_0006
Create Date: 2026-02-24 12:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20260224_0007"
down_revision = "20260221_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "canonical_insight_records",
        sa.Column(
            "role",
            sa.String(length=64),
            nullable=True,
            comment="Performance role context (e.g., organization, team, rep)",
        ),
    )
    op.create_index(
        "ix_canonical_insight_records_role",
        "canonical_insight_records",
        ["role"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_canonical_insight_records_role",
        table_name="canonical_insight_records",
    )
    op.drop_column("canonical_insight_records", "role")
