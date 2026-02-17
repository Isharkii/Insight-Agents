"""
Typed DTOs used by repository upload/storage flows.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class UploadFileInput:
    """
    Input payload for storing one uploaded dataset file.
    """

    client_id: uuid.UUID
    dataset_name: str
    file_name: str
    content: bytes
    content_type: str | None = None
    source_type: str = "upload"
    file_meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class StoredFileMetadata:
    """
    Metadata produced by the storage backend after saving a file.
    """

    file_name: str
    storage_path: str
    mime_type: str | None
    file_size_bytes: int
    checksum: str
    stored_at: datetime


@dataclass(frozen=True)
class DatasetBulkCreate:
    """
    Normalized dataset record used for bulk insert operations.
    """

    client_id: uuid.UUID
    dataset_name: str
    source_type: str
    stored_file: StoredFileMetadata
    status: str
    file_meta: dict[str, Any] | None = None
    schema_meta: dict[str, Any] | None = None
    row_count: int | None = None
    dataset_id: uuid.UUID = field(default_factory=uuid.uuid4)
