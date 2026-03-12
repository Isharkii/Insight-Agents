"""
risk/repository.py

SQLAlchemy model and repository for Business Risk Score persistence.
Alembic-ready declarative style. No scoring or business logic.

Base is imported from db.base (the project-wide shared declarative base).
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, UniqueConstraint, func, select
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, Session, mapped_column

from db.base import Base
from db.repositories.entity_scope import normalize_tenant_id, resolve_entity_scope


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BusinessRiskScore(Base):
    """Persistent record of a computed Business Risk Index score.

    Stores pre-computed results produced externally by BusinessRiskModel.
    Contains no scoring or business logic.

    Indexes:
        - tenant_id, entity_id (individual)
        - entity_name (individual, backward compatibility lookup)
        - period_end (individual)
        - (tenant_id, entity_id, period_end) composite
        - (tenant_id, entity_name, period_end) fallback composite
    """

    __tablename__ = "business_risk_scores"

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "entity_id",
            "period_end",
            name="uq_brs_tenant_entity_period",
        ),
        Index("ix_brs_tenant_entity_period", "tenant_id", "entity_id", "period_end"),
        Index(
            "ix_brs_tenant_entity_name_period",
            "tenant_id",
            "entity_name",
            "period_end",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    tenant_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        server_default="legacy",
    )
    entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenant_entities.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    entity_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )
    period_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    risk_score: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    risk_metadata: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------

class RiskRepository:
    """Data access layer for BusinessRiskScore records.

    All methods accept an active SQLAlchemy Session and operate within
    the caller's transaction boundary. No commits or rollbacks are
    issued internally — transaction control belongs to the caller.
    """

    def save_risk_score(
        self,
        session: Session,
        entity_name: str,
        period_end: datetime,
        risk_score: int,
        risk_metadata: Optional[dict] = None,
        *,
        tenant_id: str = "legacy",
        entity_id: uuid.UUID | None = None,
    ) -> BusinessRiskScore:
        """Persist a new BusinessRiskScore record within the given session.

        Adds the instance to the session without flushing or committing.
        The caller is responsible for transaction lifecycle management.

        Args:
            session: Active SQLAlchemy session.
            entity_name: Identifier for the scored business entity.
            period_end: Timestamp marking the end of the scored period.
            risk_score: Integer risk index in [0, 100].
            risk_metadata: Optional dict of raw inputs or diagnostics.
            tenant_id: Owning tenant identifier for isolation.
            entity_id: Optional stable entity identifier for exact scope.

        Returns:
            The newly created, session-tracked BusinessRiskScore instance.
        """
        scope = resolve_entity_scope(
            session,
            tenant_id=tenant_id,
            entity_name=entity_name,
            entity_id=entity_id,
            create_if_missing=True,
        )
        record = BusinessRiskScore(
            tenant_id=scope.tenant_id,
            entity_id=scope.entity_id,
            entity_name=scope.entity_name,
            period_end=period_end,
            risk_score=risk_score,
            risk_metadata=risk_metadata,
        )
        session.add(record)
        return record

    def get_latest_risk(
        self,
        session: Session,
        entity_name: str,
        *,
        tenant_id: str = "legacy",
        entity_id: uuid.UUID | None = None,
    ) -> Optional[BusinessRiskScore]:
        """Retrieve the most recent risk score for a given entity.

        Orders by period_end descending, then created_at descending
        as a tiebreaker when multiple records share the same period_end.

        Args:
            session: Active SQLAlchemy session.
            entity_name: Identifier for the business entity to query.
            tenant_id: Owning tenant identifier for isolation.
            entity_id: Optional stable entity identifier for exact scope.

        Returns:
            The latest BusinessRiskScore instance, or None if absent.
        """
        stmt = select(BusinessRiskScore).where(
            BusinessRiskScore.tenant_id == normalize_tenant_id(tenant_id)
        )
        if entity_id is not None:
            stmt = stmt.where(BusinessRiskScore.entity_id == entity_id)
        else:
            stmt = stmt.where(BusinessRiskScore.entity_name == entity_name)
        stmt = stmt.order_by(
            BusinessRiskScore.period_end.desc(),
            BusinessRiskScore.created_at.desc(),
        ).limit(1)
        return session.scalars(stmt).first()
