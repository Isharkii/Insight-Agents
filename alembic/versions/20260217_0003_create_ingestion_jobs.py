"""create ingestion_jobs table

Revision ID: 20260217_0003
Revises: 20260217_0002
Create Date: 2026-02-17 17:15:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20260217_0003"
down_revision = "20260217_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ingestion_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("job_type", sa.String(length=50), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("request_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("result_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_ingestion_jobs_created_at", "ingestion_jobs", ["created_at"], unique=False)
    op.create_index("ix_ingestion_jobs_job_type", "ingestion_jobs", ["job_type"], unique=False)
    op.create_index(
        "ix_ingestion_jobs_job_type_status",
        "ingestion_jobs",
        ["job_type", "status"],
        unique=False,
    )
    op.create_index("ix_ingestion_jobs_status", "ingestion_jobs", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_ingestion_jobs_status", table_name="ingestion_jobs")
    op.drop_index("ix_ingestion_jobs_job_type_status", table_name="ingestion_jobs")
    op.drop_index("ix_ingestion_jobs_job_type", table_name="ingestion_jobs")
    op.drop_index("ix_ingestion_jobs_created_at", table_name="ingestion_jobs")
    op.drop_table("ingestion_jobs")
