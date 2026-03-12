"""add tenant/entity scoping to analytical persistence tables

Revision ID: 20260309_0010
Revises: 20260302_0009
Create Date: 2026-03-09 00:00:00
"""

from __future__ import annotations

import uuid

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260309_0010"
down_revision = "20260302_0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tenant_entities",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("entity_key", sa.String(length=255), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=False),
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
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "tenant_id",
            "entity_key",
            name="uq_tenant_entities_tenant_entity_key",
        ),
    )
    op.create_index("ix_tenant_entities_tenant_id", "tenant_entities", ["tenant_id"])
    op.create_index(
        "ix_tenant_entities_tenant_display_name",
        "tenant_entities",
        ["tenant_id", "display_name"],
    )

    op.add_column(
        "computed_kpis",
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("'legacy'"),
        ),
    )
    op.add_column(
        "computed_kpis",
        sa.Column("entity_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "forecast_metric",
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("'legacy'"),
        ),
    )
    op.add_column(
        "forecast_metric",
        sa.Column("entity_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "business_risk_scores",
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("'legacy'"),
        ),
    )
    op.add_column(
        "business_risk_scores",
        sa.Column("entity_id", postgresql.UUID(as_uuid=True), nullable=True),
    )

    conn = op.get_bind()
    tenant_entities_table = sa.table(
        "tenant_entities",
        sa.column("id", postgresql.UUID(as_uuid=True)),
        sa.column("tenant_id", sa.String(length=64)),
        sa.column("entity_key", sa.String(length=255)),
        sa.column("display_name", sa.String(length=255)),
    )
    names: set[str] = set()
    for table_name in ("computed_kpis", "forecast_metric", "business_risk_scores"):
        rows = conn.execute(
            sa.text(
                f"""
                SELECT DISTINCT entity_name
                FROM {table_name}
                WHERE entity_name IS NOT NULL
                  AND btrim(entity_name) <> ''
                """
            )
        )
        for row in rows:
            names.add(str(row[0]).strip())

    if names:
        op.bulk_insert(
            tenant_entities_table,
            [
                {
                    "id": uuid.uuid4(),
                    "tenant_id": "legacy",
                    "entity_key": name,
                    "display_name": name,
                }
                for name in sorted(names)
            ],
        )

    op.execute(
        sa.text(
            """
            UPDATE computed_kpis AS ck
            SET entity_id = te.id
            FROM tenant_entities AS te
            WHERE te.tenant_id = ck.tenant_id
              AND te.entity_key = ck.entity_name
            """
        )
    )
    op.execute(
        sa.text(
            """
            UPDATE forecast_metric AS fm
            SET entity_id = te.id
            FROM tenant_entities AS te
            WHERE te.tenant_id = fm.tenant_id
              AND te.entity_key = fm.entity_name
            """
        )
    )
    op.execute(
        sa.text(
            """
            UPDATE business_risk_scores AS brs
            SET entity_id = te.id
            FROM tenant_entities AS te
            WHERE te.tenant_id = brs.tenant_id
              AND te.entity_key = brs.entity_name
            """
        )
    )

    op.alter_column(
        "computed_kpis",
        "entity_id",
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=False,
    )
    op.alter_column(
        "forecast_metric",
        "entity_id",
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=False,
    )
    op.alter_column(
        "business_risk_scores",
        "entity_id",
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=False,
    )

    op.create_foreign_key(
        "fk_computed_kpis_entity_id_tenant_entities",
        "computed_kpis",
        "tenant_entities",
        ["entity_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_foreign_key(
        "fk_forecast_metric_entity_id_tenant_entities",
        "forecast_metric",
        "tenant_entities",
        ["entity_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_foreign_key(
        "fk_business_risk_scores_entity_id_tenant_entities",
        "business_risk_scores",
        "tenant_entities",
        ["entity_id"],
        ["id"],
        ondelete="RESTRICT",
    )

    op.drop_constraint("uq_computed_kpis_entity_period", "computed_kpis", type_="unique")
    op.drop_constraint(
        "uq_forecast_metric_entity_metric_period",
        "forecast_metric",
        type_="unique",
    )
    op.drop_constraint("uq_brs_entity_period", "business_risk_scores", type_="unique")

    op.create_unique_constraint(
        "uq_computed_kpis_tenant_entity_period",
        "computed_kpis",
        ["tenant_id", "entity_id", "period_start", "period_end"],
    )
    op.create_unique_constraint(
        "uq_forecast_metric_tenant_entity_metric_period",
        "forecast_metric",
        ["tenant_id", "entity_id", "metric_name", "period_end"],
    )
    op.create_unique_constraint(
        "uq_brs_tenant_entity_period",
        "business_risk_scores",
        ["tenant_id", "entity_id", "period_end"],
    )

    op.drop_index("ix_computed_kpis_entity_period_start", table_name="computed_kpis")
    op.drop_index("ix_forecast_metric_entity_metric_period", table_name="forecast_metric")
    op.drop_index("ix_brs_entity_period", table_name="business_risk_scores")

    op.create_index("ix_computed_kpis_tenant_id", "computed_kpis", ["tenant_id"])
    op.create_index("ix_computed_kpis_entity_id", "computed_kpis", ["entity_id"])
    op.create_index(
        "ix_computed_kpis_tenant_entity_period_start",
        "computed_kpis",
        ["tenant_id", "entity_id", "period_start"],
    )
    op.create_index(
        "ix_computed_kpis_tenant_entity_name_period_start",
        "computed_kpis",
        ["tenant_id", "entity_name", "period_start"],
    )
    op.create_index(
        "ix_computed_kpis_tenant_period_start",
        "computed_kpis",
        ["tenant_id", "period_start"],
    )

    op.create_index("ix_forecast_metric_tenant_id", "forecast_metric", ["tenant_id"])
    op.create_index("ix_forecast_metric_entity_id", "forecast_metric", ["entity_id"])
    op.create_index(
        "ix_forecast_metric_tenant_entity_metric_period",
        "forecast_metric",
        ["tenant_id", "entity_id", "metric_name", "period_end"],
    )
    op.create_index(
        "ix_forecast_metric_tenant_entity_name_metric_period",
        "forecast_metric",
        ["tenant_id", "entity_name", "metric_name", "period_end"],
    )

    op.create_index(
        "ix_business_risk_scores_tenant_id",
        "business_risk_scores",
        ["tenant_id"],
    )
    op.create_index(
        "ix_business_risk_scores_entity_id",
        "business_risk_scores",
        ["entity_id"],
    )
    op.create_index(
        "ix_brs_tenant_entity_period",
        "business_risk_scores",
        ["tenant_id", "entity_id", "period_end"],
    )
    op.create_index(
        "ix_brs_tenant_entity_name_period",
        "business_risk_scores",
        ["tenant_id", "entity_name", "period_end"],
    )


def downgrade() -> None:
    op.drop_index("ix_brs_tenant_entity_period", table_name="business_risk_scores")
    op.drop_index(
        "ix_brs_tenant_entity_name_period",
        table_name="business_risk_scores",
    )
    op.drop_index("ix_business_risk_scores_entity_id", table_name="business_risk_scores")
    op.drop_index("ix_business_risk_scores_tenant_id", table_name="business_risk_scores")
    op.drop_index(
        "ix_forecast_metric_tenant_entity_name_metric_period",
        table_name="forecast_metric",
    )
    op.drop_index(
        "ix_forecast_metric_tenant_entity_metric_period",
        table_name="forecast_metric",
    )
    op.drop_index("ix_forecast_metric_entity_id", table_name="forecast_metric")
    op.drop_index("ix_forecast_metric_tenant_id", table_name="forecast_metric")
    op.drop_index("ix_computed_kpis_tenant_period_start", table_name="computed_kpis")
    op.drop_index(
        "ix_computed_kpis_tenant_entity_name_period_start",
        table_name="computed_kpis",
    )
    op.drop_index(
        "ix_computed_kpis_tenant_entity_period_start",
        table_name="computed_kpis",
    )
    op.drop_index("ix_computed_kpis_entity_id", table_name="computed_kpis")
    op.drop_index("ix_computed_kpis_tenant_id", table_name="computed_kpis")

    op.drop_constraint(
        "uq_brs_tenant_entity_period",
        "business_risk_scores",
        type_="unique",
    )
    op.drop_constraint(
        "uq_forecast_metric_tenant_entity_metric_period",
        "forecast_metric",
        type_="unique",
    )
    op.drop_constraint(
        "uq_computed_kpis_tenant_entity_period",
        "computed_kpis",
        type_="unique",
    )

    op.create_unique_constraint(
        "uq_brs_entity_period",
        "business_risk_scores",
        ["entity_name", "period_end"],
    )
    op.create_unique_constraint(
        "uq_forecast_metric_entity_metric_period",
        "forecast_metric",
        ["entity_name", "metric_name", "period_end"],
    )
    op.create_unique_constraint(
        "uq_computed_kpis_entity_period",
        "computed_kpis",
        ["entity_name", "period_start", "period_end"],
    )

    op.create_index("ix_brs_entity_period", "business_risk_scores", ["entity_name", "period_end"])
    op.create_index(
        "ix_forecast_metric_entity_metric_period",
        "forecast_metric",
        ["entity_name", "metric_name", "period_end"],
    )
    op.create_index(
        "ix_computed_kpis_entity_period_start",
        "computed_kpis",
        ["entity_name", "period_start"],
    )

    op.drop_constraint(
        "fk_business_risk_scores_entity_id_tenant_entities",
        "business_risk_scores",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_forecast_metric_entity_id_tenant_entities",
        "forecast_metric",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_computed_kpis_entity_id_tenant_entities",
        "computed_kpis",
        type_="foreignkey",
    )

    op.drop_column("business_risk_scores", "entity_id")
    op.drop_column("business_risk_scores", "tenant_id")
    op.drop_column("forecast_metric", "entity_id")
    op.drop_column("forecast_metric", "tenant_id")
    op.drop_column("computed_kpis", "entity_id")
    op.drop_column("computed_kpis", "tenant_id")

    op.drop_index("ix_tenant_entities_tenant_display_name", table_name="tenant_entities")
    op.drop_index("ix_tenant_entities_tenant_id", table_name="tenant_entities")
    op.drop_table("tenant_entities")
