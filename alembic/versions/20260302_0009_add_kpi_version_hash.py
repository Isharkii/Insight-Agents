"""Add analytics_version and dataset_hash to computed_kpis.

Revision ID: 20260302_0009
Revises: 20260226_0008
Create Date: 2026-03-02

Enables deterministic recomputation: the analytics guard can skip
computation only when both the pipeline version AND the underlying
dataset hash match.  Existing rows default to NULL which forces
recomputation on the next run.
"""

from alembic import op
import sqlalchemy as sa

revision = "20260302_0009"
down_revision = "20260226_0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "computed_kpis",
        sa.Column("analytics_version", sa.Integer(), nullable=True),
    )
    op.add_column(
        "computed_kpis",
        sa.Column("dataset_hash", sa.String(64), nullable=True),
    )
    op.create_index(
        "ix_computed_kpis_entity_version",
        "computed_kpis",
        ["entity_name", "analytics_version"],
    )


def downgrade() -> None:
    op.drop_index("ix_computed_kpis_entity_version", table_name="computed_kpis")
    op.drop_column("computed_kpis", "dataset_hash")
    op.drop_column("computed_kpis", "analytics_version")
