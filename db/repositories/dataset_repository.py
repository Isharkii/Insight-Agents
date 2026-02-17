"""
Dataset repository responsible for DB writes and lookup operations.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from db.models.client import Client
from db.models.dataset import Dataset, DatasetStatus
from db.repositories.errors import ClientInactiveError, ClientNotFoundError
from db.repositories.types import DatasetBulkCreate, StoredFileMetadata


class DatasetRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def ensure_client_exists(self, client_id: uuid.UUID, *, active_only: bool = True) -> Client:
        client = self._session.get(Client, client_id)
        if client is None:
            raise ClientNotFoundError(f"Client not found: {client_id}")
        if active_only and not client.is_active:
            raise ClientInactiveError(f"Client is inactive: {client_id}")
        return client

    def create_dataset_reference(
        self,
        *,
        client_id: uuid.UUID,
        dataset_name: str,
        source_type: str,
        stored_file: StoredFileMetadata,
        status: str = DatasetStatus.PENDING,
        file_meta: dict | None = None,
    ) -> Dataset:
        dataset = Dataset(
            client_id=client_id,
            name=dataset_name,
            source_type=source_type,
            status=status,
            file_name=stored_file.file_name,
            file_path=stored_file.storage_path,
            mime_type=stored_file.mime_type,
            file_size_bytes=stored_file.file_size_bytes,
            checksum=stored_file.checksum,
            file_meta=file_meta,
        )
        self._session.add(dataset)
        return dataset

    def bulk_create_dataset_references(
        self,
        records: Sequence[DatasetBulkCreate],
        *,
        batch_size: int = 500,
    ) -> list[uuid.UUID]:
        """
        Efficient bulk insert strategy for dataset metadata rows.

        Uses PostgreSQL INSERT with chunking instead of ORM per-row add/flush.
        """

        if not records:
            return []

        created_ids: list[uuid.UUID] = []
        for chunk_start in range(0, len(records), batch_size):
            chunk = records[chunk_start : chunk_start + batch_size]
            values = [
                {
                    "id": record.dataset_id,
                    "client_id": record.client_id,
                    "name": record.dataset_name,
                    "source_type": record.source_type,
                    "status": record.status,
                    "file_name": record.stored_file.file_name,
                    "file_path": record.stored_file.storage_path,
                    "mime_type": record.stored_file.mime_type,
                    "file_size_bytes": record.stored_file.file_size_bytes,
                    "checksum": record.stored_file.checksum,
                    "row_count": record.row_count,
                    "schema_meta": record.schema_meta,
                    "file_meta": record.file_meta,
                }
                for record in chunk
            ]
            self._session.execute(insert(Dataset), values)
            created_ids.extend(record.dataset_id for record in chunk)

        return created_ids

    def list_datasets_for_client(self, client_id: uuid.UUID, *, limit: int = 100) -> list[Dataset]:
        stmt = (
            select(Dataset)
            .where(Dataset.client_id == client_id)
            .order_by(Dataset.created_at.desc())
            .limit(limit)
        )
        return list(self._session.scalars(stmt).all())
