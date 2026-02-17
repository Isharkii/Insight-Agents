"""
Storage backend abstractions for uploaded dataset files.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from mimetypes import guess_type
from pathlib import Path
from typing import Protocol

from db.repositories.errors import FileStorageError
from db.repositories.types import StoredFileMetadata


class FileStorageBackend(Protocol):
    """
    Abstract storage backend used by upload repository.
    """

    def save(
        self,
        *,
        client_id: uuid.UUID,
        file_name: str,
        content: bytes,
        content_type: str | None = None,
    ) -> StoredFileMetadata:
        ...

    def delete(self, *, storage_path: str) -> None:
        ...


def _sanitize_file_name(file_name: str) -> str:
    safe_name = Path(file_name).name.strip()
    if not safe_name:
        raise FileStorageError("Invalid file name.")
    return safe_name


class LocalFileStorage:
    """
    Local filesystem storage backend.

    This is intentionally simple and can be replaced with object storage later.
    """

    def __init__(self, root_dir: str | Path = "data/uploads") -> None:
        self._root_dir = Path(root_dir)

    def save(
        self,
        *,
        client_id: uuid.UUID,
        file_name: str,
        content: bytes,
        content_type: str | None = None,
    ) -> StoredFileMetadata:
        safe_file_name = _sanitize_file_name(file_name)
        stored_at = datetime.now(timezone.utc)

        relative_path = (
            Path(str(client_id))
            / stored_at.strftime("%Y")
            / stored_at.strftime("%m")
            / f"{uuid.uuid4().hex}_{safe_file_name}"
        )
        absolute_path = self._root_dir / relative_path
        absolute_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = absolute_path.with_suffix(f"{absolute_path.suffix}.tmp")
        try:
            with tmp_path.open("wb") as handle:
                handle.write(content)
            tmp_path.replace(absolute_path)
        except OSError as exc:
            raise FileStorageError("Failed to write uploaded file to storage.") from exc
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

        checksum = hashlib.sha256(content).hexdigest()
        guessed_mime = guess_type(safe_file_name)[0]
        mime_type = content_type or guessed_mime

        return StoredFileMetadata(
            file_name=safe_file_name,
            storage_path=relative_path.as_posix(),
            mime_type=mime_type,
            file_size_bytes=len(content),
            checksum=checksum,
            stored_at=stored_at,
        )

    def delete(self, *, storage_path: str) -> None:
        target = self._root_dir / Path(storage_path)
        if not target.exists():
            return
        try:
            target.unlink()
        except OSError as exc:
            raise FileStorageError("Failed to delete uploaded file from storage.") from exc
