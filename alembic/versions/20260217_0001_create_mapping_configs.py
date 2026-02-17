"""create mapping_configs table

Revision ID: 20260217_0001
Revises:
Create Date: 2026-02-17 16:20:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20260217_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "mapping_configs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("client_name", sa.String(length=255), nullable=True),
        sa.Column("field_mapping_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("alias_overrides_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("notes", sa.String(length=500), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", "client_name", name="uq_mapping_configs_name_client_name"),
    )
    op.create_index("ix_mapping_configs_name", "mapping_configs", ["name"], unique=False)
    op.create_index("ix_mapping_configs_client_name", "mapping_configs", ["client_name"], unique=False)
    op.create_index("ix_mapping_configs_is_active", "mapping_configs", ["is_active"], unique=False)
    op.create_index(
        "ix_mapping_configs_client_active",
        "mapping_configs",
        ["client_name", "is_active"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_mapping_configs_client_active", table_name="mapping_configs")
    op.drop_index("ix_mapping_configs_is_active", table_name="mapping_configs")
    op.drop_index("ix_mapping_configs_client_name", table_name="mapping_configs")
    op.drop_index("ix_mapping_configs_name", table_name="mapping_configs")
    op.drop_table("mapping_configs")
