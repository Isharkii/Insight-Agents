"""
app/repositories/mapping_config_repository.py

Persistence helpers for schema mapping configurations.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models.mapping_config import MappingConfig


class MappingConfigRepository:
    """
    Repository for CRUD-like operations on mapping configurations.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def get_active(
        self,
        *,
        name: str | None = None,
        client_name: str | None = None,
    ) -> MappingConfig | None:
        """
        Resolve one active mapping config by optional name/client scope.
        """

        stmt = select(MappingConfig).where(MappingConfig.is_active.is_(True))
        if name:
            stmt = stmt.where(MappingConfig.name == name.strip())
        if client_name:
            stmt = stmt.where(MappingConfig.client_name == client_name.strip())
        stmt = stmt.order_by(MappingConfig.updated_at.desc())
        return self._session.execute(stmt).scalars().first()

    def save(
        self,
        *,
        name: str,
        field_mapping: dict[str, str],
        client_name: str | None = None,
        alias_overrides: dict[str, list[str]] | None = None,
        notes: str | None = None,
        metadata_json: dict[str, Any] | None = None,
        is_active: bool = True,
    ) -> MappingConfig:
        """
        Insert or update mapping config keyed by (name, client_name).
        """

        normalized_name = name.strip()
        normalized_client = client_name.strip() if client_name else None

        stmt = select(MappingConfig).where(MappingConfig.name == normalized_name)
        if normalized_client is None:
            stmt = stmt.where(MappingConfig.client_name.is_(None))
        else:
            stmt = stmt.where(MappingConfig.client_name == normalized_client)
        existing = self._session.execute(stmt).scalars().first()

        if existing is None:
            existing = MappingConfig(
                name=normalized_name,
                client_name=normalized_client,
                field_mapping_json=field_mapping,
                alias_overrides_json=alias_overrides,
                notes=notes,
                metadata_json=metadata_json,
                is_active=is_active,
            )
            self._session.add(existing)
        else:
            existing.field_mapping_json = field_mapping
            existing.alias_overrides_json = alias_overrides
            existing.notes = notes
            existing.metadata_json = metadata_json
            existing.is_active = is_active

        self._session.flush()
        return existing
