"""
Repository-layer exceptions for upload/storage flows.
"""

from __future__ import annotations


class UploadRepositoryError(Exception):
    """Base exception for upload repository failures."""


class UploadValidationError(UploadRepositoryError):
    """Raised when upload payload validation fails."""


class FileStorageError(UploadRepositoryError):
    """Raised when storing or deleting uploaded files fails."""


class ClientNotFoundError(UploadRepositoryError):
    """Raised when a referenced client does not exist."""


class ClientInactiveError(UploadRepositoryError):
    """Raised when a referenced client is not active."""


class DatasetPersistenceError(UploadRepositoryError):
    """Raised when dataset metadata persistence fails."""
