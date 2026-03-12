"""
db/repositories/entity_scope.py

Tenant/entity scope resolution primitives for analytics repositories.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models.tenant_entity import TenantEntity

_DEFAULT_TENANT_ID = "legacy"


@dataclass(frozen=True)
class EntityScope:
    tenant_id: str
    entity_id: uuid.UUID
    entity_name: str


def normalize_tenant_id(value: str | None) -> str:
    text = str(value or "").strip()
    if text:
        return text
    return _DEFAULT_TENANT_ID


def resolve_entity_scope(
    session: Session,
    *,
    tenant_id: str | None,
    entity_name: str | None,
    entity_id: uuid.UUID | None = None,
    create_if_missing: bool = True,
) -> EntityScope:
    """Resolve `(tenant_id, entity_id, entity_name)` for tenant-scoped writes/reads."""
    tenant = normalize_tenant_id(tenant_id)
    resolved_name = str(entity_name or "").strip()

    if entity_id is not None:
        row = session.get(TenantEntity, entity_id)
        if row is None:
            raise ValueError(f"Unknown entity_id: {entity_id}")
        if row.tenant_id != tenant:
            raise ValueError(
                f"entity_id {entity_id} does not belong to tenant_id={tenant!r}"
            )
        return EntityScope(
            tenant_id=row.tenant_id,
            entity_id=row.id,
            entity_name=resolved_name or row.entity_key,
        )

    if not resolved_name:
        raise ValueError("entity_name or entity_id is required.")

    existing = session.scalars(
        select(TenantEntity).where(
            TenantEntity.tenant_id == tenant,
            TenantEntity.entity_key == resolved_name,
        )
    ).first()
    if existing is not None:
        return EntityScope(
            tenant_id=existing.tenant_id,
            entity_id=existing.id,
            entity_name=resolved_name,
        )

    if not create_if_missing:
        raise ValueError(
            f"Unknown entity_name={resolved_name!r} for tenant_id={tenant!r}"
        )

    created = TenantEntity(
        id=uuid.uuid4(),
        tenant_id=tenant,
        entity_key=resolved_name,
        display_name=resolved_name,
    )
    session.add(created)
    session.flush()
    return EntityScope(
        tenant_id=tenant,
        entity_id=created.id,
        entity_name=resolved_name,
    )
