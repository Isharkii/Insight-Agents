"""add unique constraints to forecast_metric and business_risk_scores

Revision ID: 20260221_0006
Revises: 20260221_0005
Create Date: 2026-02-21 00:00:00
"""

from __future__ import annotations

from alembic import op

revision = "20260221_0006"
down_revision = "20260221_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_forecast_metric_entity_metric_period",
        "forecast_metric",
        ["entity_name", "metric_name", "period_end"],
    )
    op.create_unique_constraint(
        "uq_brs_entity_period",
        "business_risk_scores",
        ["entity_name", "period_end"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_brs_entity_period",
        "business_risk_scores",
        type_="unique",
    )
    op.drop_constraint(
        "uq_forecast_metric_entity_metric_period",
        "forecast_metric",
        type_="unique",
    )
