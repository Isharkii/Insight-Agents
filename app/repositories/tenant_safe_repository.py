"""
Tenant-safe repository wrappers for analytical persistence.

This module demonstrates the target access pattern:
1) every query/write is constrained by tenant_id
2) every analytical row references stable entity_id
3) entity_name is treated as denormalized display data only
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from db.models.computed_kpi import ComputedKPI
from db.repositories.kpi_repository import KPIRepository
from forecast.repository import ForecastMetric, ForecastRepository
from risk.repository import BusinessRiskScore, RiskRepository


@dataclass(frozen=True)
class TenantContext:
    tenant_id: str


@dataclass(frozen=True)
class EntityContext:
    entity_name: str
    entity_id: uuid.UUID | None = None


class TenantScopedAnalyticsRepository:
    """Unified tenant-safe facade for KPI, forecast, and risk persistence."""

    def __init__(self, session: Session, *, tenant: TenantContext) -> None:
        self._session = session
        self._tenant = tenant
        self._kpis = KPIRepository(session)
        self._forecasts = ForecastRepository(session)
        self._risks = RiskRepository()

    @property
    def tenant_id(self) -> str:
        return str(self._tenant.tenant_id or "").strip() or "legacy"

    def save_kpis(
        self,
        *,
        entity: EntityContext,
        period_start: datetime,
        period_end: datetime,
        computed_kpis: dict[str, Any],
        analytics_version: int | None = None,
        dataset_hash: str | None = None,
    ) -> ComputedKPI:
        return self._kpis.save_kpi(
            tenant_id=self.tenant_id,
            entity_id=entity.entity_id,
            entity_name=entity.entity_name,
            period_start=period_start,
            period_end=period_end,
            computed_kpis=computed_kpis,
            analytics_version=analytics_version,
            dataset_hash=dataset_hash,
        )

    def get_latest_forecast(
        self,
        *,
        entity: EntityContext,
        metric_name: str,
    ) -> ForecastMetric | None:
        return self._forecasts.get_latest_forecast(
            entity_name=entity.entity_name,
            entity_id=entity.entity_id,
            tenant_id=self.tenant_id,
            metric_name=metric_name,
        )

    def save_forecast(
        self,
        *,
        entity: EntityContext,
        metric_name: str,
        period_end: datetime,
        forecast_data: dict[str, Any],
    ) -> ForecastMetric:
        return self._forecasts.save_forecast(
            entity_name=entity.entity_name,
            entity_id=entity.entity_id,
            tenant_id=self.tenant_id,
            metric_name=metric_name,
            period_end=period_end,
            forecast_data=forecast_data,
        )

    def save_risk_score(
        self,
        *,
        entity: EntityContext,
        period_end: datetime,
        risk_score: int,
        risk_metadata: dict[str, Any] | None = None,
    ) -> BusinessRiskScore:
        return self._risks.save_risk_score(
            session=self._session,
            entity_name=entity.entity_name,
            entity_id=entity.entity_id,
            tenant_id=self.tenant_id,
            period_end=period_end,
            risk_score=risk_score,
            risk_metadata=risk_metadata,
        )
