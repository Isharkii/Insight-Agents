"""
Validation helpers for upload repository flows.
"""

from __future__ import annotations

import os
from pathlib import Path

from db.repositories.errors import UploadValidationError
from db.repositories.types import UploadFileInput

ALLOWED_EXTENSIONS = {".csv", ".xls", ".xlsx"}
ALLOWED_CONTENT_TYPES = {
    "text/csv",
    "application/csv",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/octet-stream",
}


def _max_upload_size_bytes() -> int:
    value = os.getenv("UPLOAD_MAX_BYTES")
    if value is None:
        return 50 * 1024 * 1024
    try:
        return int(value)
    except ValueError:
        return 50 * 1024 * 1024


def validate_upload_payload(payload: UploadFileInput) -> None:
    """
    Validate file upload payload before storage and DB persistence.
    """

    if not payload.dataset_name or not payload.dataset_name.strip():
        raise UploadValidationError("dataset_name is required.")

    if not payload.file_name or not payload.file_name.strip():
        raise UploadValidationError("file_name is required.")

    extension = Path(payload.file_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise UploadValidationError(
            f"Unsupported file type '{extension}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}."
        )

    if payload.content_type and payload.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise UploadValidationError(
            f"Unsupported content_type '{payload.content_type}'."
        )

    if not payload.content:
        raise UploadValidationError("Uploaded file content is empty.")

    if len(payload.content) > _max_upload_size_bytes():
        raise UploadValidationError("Uploaded file exceeds configured size limit.")
