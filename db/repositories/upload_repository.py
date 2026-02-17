"""
Upload repository orchestrating file storage and dataset-reference persistence.

Storage only: does not run metric or insight processing.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Sequence
from typing import Protocol

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from db.models.dataset import Dataset, DatasetStatus
from db.repositories.dataset_repository import DatasetRepository
from db.repositories.errors import DatasetPersistenceError, FileStorageError
from db.repositories.storage import FileStorageBackend, LocalFileStorage
from db.repositories.types import DatasetBulkCreate, UploadFileInput
from db.repositories.validators import validate_upload_payload


class PostStoreHook(Protocol):
    """
    Hook point for future background processing integration.
    """

    def on_dataset_stored(
        self,
        *,
        dataset_id: uuid.UUID,
        client_id: uuid.UUID,
        storage_path: str,
    ) -> None:
        ...


class NoOpPostStoreHook:
    def on_dataset_stored(
        self,
        *,
        dataset_id: uuid.UUID,
        client_id: uuid.UUID,
        storage_path: str,
    ) -> None:
        return None


class UploadRepository:
    """
    Repository layer entrypoint for upload->storage->dataset-reference flow.
    """

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session] | None = None,
        storage_backend: FileStorageBackend | None = None,
        post_store_hook: PostStoreHook | None = None,
    ) -> None:
        storage_dir = os.getenv("UPLOAD_STORAGE_DIR", "data/uploads")
        if session_factory is None:
            from db.session import SessionLocal

            self._session_factory = SessionLocal
        else:
            self._session_factory = session_factory
        self._storage_backend = storage_backend or LocalFileStorage(storage_dir)
        self._post_store_hook = post_store_hook or NoOpPostStoreHook()

    def store_upload(self, payload: UploadFileInput) -> Dataset:
        """
        Store one uploaded CSV/Excel file and persist a dataset DB reference.

        Transaction safety:
        - DB insert runs in a transaction.
        - On DB failure, stored file is deleted as compensating action.
        """

        validate_upload_payload(payload)

        stored = self._storage_backend.save(
            client_id=payload.client_id,
            file_name=payload.file_name,
            content=payload.content,
            content_type=payload.content_type,
        )

        dataset: Dataset | None = None
        try:
            with self._session_factory() as session:
                repo = DatasetRepository(session)
                with session.begin():
                    repo.ensure_client_exists(payload.client_id, active_only=True)
                    dataset = repo.create_dataset_reference(
                        client_id=payload.client_id,
                        dataset_name=payload.dataset_name,
                        source_type=payload.source_type,
                        stored_file=stored,
                        status=DatasetStatus.PENDING,
                        file_meta=payload.file_meta,
                    )
                    session.flush()
                    session.refresh(dataset)
        except SQLAlchemyError as exc:
            self._delete_stored_file_quietly(stored.storage_path)
            raise DatasetPersistenceError("Failed to persist dataset metadata.") from exc
        except Exception:
            self._delete_stored_file_quietly(stored.storage_path)
            raise

        if dataset is None:
            self._delete_stored_file_quietly(stored.storage_path)
            raise DatasetPersistenceError("Dataset was not created.")

        self._trigger_hook(
            dataset_id=dataset.id,
            client_id=payload.client_id,
            storage_path=stored.storage_path,
        )
        return dataset

    def store_uploads_bulk(
        self,
        payloads: Sequence[UploadFileInput],
        *,
        batch_size: int = 500,
    ) -> list[uuid.UUID]:
        """
        Store multiple uploads with efficient chunked bulk DB insert.
        """

        if not payloads:
            return []

        stored_records: list[tuple[UploadFileInput, DatasetBulkCreate]] = []
        try:
            for payload in payloads:
                validate_upload_payload(payload)
                stored = self._storage_backend.save(
                    client_id=payload.client_id,
                    file_name=payload.file_name,
                    content=payload.content,
                    content_type=payload.content_type,
                )
                record = DatasetBulkCreate(
                    client_id=payload.client_id,
                    dataset_name=payload.dataset_name,
                    source_type=payload.source_type,
                    stored_file=stored,
                    status=DatasetStatus.PENDING,
                    file_meta=payload.file_meta,
                )
                stored_records.append((payload, record))

            with self._session_factory() as session:
                repo = DatasetRepository(session)
                with session.begin():
                    checked_clients: set[uuid.UUID] = set()
                    for payload, _ in stored_records:
                        if payload.client_id in checked_clients:
                            continue
                        repo.ensure_client_exists(payload.client_id, active_only=True)
                        checked_clients.add(payload.client_id)

                    created_ids = repo.bulk_create_dataset_references(
                        [record for _, record in stored_records],
                        batch_size=batch_size,
                    )
        except SQLAlchemyError as exc:
            self._cleanup_stored_records(stored_records)
            raise DatasetPersistenceError("Bulk dataset metadata insert failed.") from exc
        except Exception:
            self._cleanup_stored_records(stored_records)
            raise

        for payload, record in stored_records:
            self._trigger_hook(
                dataset_id=record.dataset_id,
                client_id=payload.client_id,
                storage_path=record.stored_file.storage_path,
            )
        return created_ids

    def _cleanup_stored_records(
        self,
        stored_records: Sequence[tuple[UploadFileInput, DatasetBulkCreate]],
    ) -> None:
        for _, record in stored_records:
            self._delete_stored_file_quietly(record.stored_file.storage_path)

    def _delete_stored_file_quietly(self, storage_path: str) -> None:
        try:
            self._storage_backend.delete(storage_path=storage_path)
        except FileStorageError:
            pass

    def _trigger_hook(
        self,
        *,
        dataset_id: uuid.UUID,
        client_id: uuid.UUID,
        storage_path: str,
    ) -> None:
        # Hook is best-effort; upload storage success should remain durable.
        try:
            self._post_store_hook.on_dataset_stored(
                dataset_id=dataset_id,
                client_id=client_id,
                storage_path=storage_path,
            )
        except Exception:
            return
