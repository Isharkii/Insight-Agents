"""
app/services/csv_ingestion_service.py

Service layer for CSV ingestion workflow orchestration.
"""

from __future__ import annotations

import csv
import io
import logging
from functools import lru_cache
from typing import Mapping

from fastapi import UploadFile
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import get_csv_ingestion_settings
from app.domain.canonical_insight import CanonicalInsightInput, IngestionSummary, RowValidationError
from app.mappers.schema_mapper import MappingResolution, SchemaMapper
from app.repositories.canonical_insight_repository import CanonicalInsightRepository
from app.repositories.mapping_config_repository import MappingConfigRepository
from app.validators.mapping_validator import MappingErrorDetail, SchemaMappingError
from app.validators.csv_validator import CSVRowValidator

logger = logging.getLogger(__name__)


class CSVHeaderValidationError(ValueError):
    """
    Raised when CSV shape/header validation fails.
    """


class CSVPersistenceError(RuntimeError):
    """
    Raised when valid rows cannot be persisted.
    """


class CSVSchemaMappingError(CSVHeaderValidationError):
    """
    Raised when CSV schema mapping resolution fails with structured details.
    """

    def __init__(self, *, message: str, errors: list[MappingErrorDetail]) -> None:
        super().__init__(message)
        self.errors = tuple(errors)

    def to_dict(self) -> dict[str, object]:
        return {
            "message": str(self),
            "errors": [
                {
                    "code": error.code,
                    "message": error.message,
                    "canonical_field": error.canonical_field,
                    "source_column": error.source_column,
                    "context": error.context,
                }
                for error in self.errors
            ],
        }


class CSVIngestionService:
    """
    Coordinates CSV parsing, mapping, validation, and persistence.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        max_validation_errors: int,
        log_validation_errors: bool,
        mapper: SchemaMapper | None = None,
        validator: CSVRowValidator | None = None,
    ) -> None:
        self._batch_size = max(1, batch_size)
        self._max_validation_errors = max(1, max_validation_errors)
        self._log_validation_errors = log_validation_errors
        self._mapper = mapper or SchemaMapper()
        self._validator = validator or CSVRowValidator()

    def ingest_csv(
        self,
        *,
        upload_file: UploadFile,
        db: Session,
        client_name: str | None = None,
        mapping_config_name: str | None = None,
        manual_mapping: Mapping[str, str] | None = None,
    ) -> IngestionSummary:
        """
        Stream a CSV file, skip invalid rows, and persist valid rows in batches.
        """

        raw_file = upload_file.file
        raw_file.seek(0)
        text_stream: io.TextIOWrapper | None = None
        repository = CanonicalInsightRepository(db)

        rows_processed = 0
        rows_failed = 0
        captured_errors: list[RowValidationError] = []
        batch: list[CanonicalInsightInput] = []

        try:
            text_stream = io.TextIOWrapper(raw_file, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(text_stream)
            headers = reader.fieldnames or []
            if not headers:
                raise CSVHeaderValidationError("CSV header row is missing.")

            mapping = self._resolve_mapping(
                db=db,
                headers=headers,
                client_name=client_name,
                mapping_config_name=mapping_config_name,
                manual_mapping=manual_mapping,
            )

            for row_number, raw_row in enumerate(reader, start=2):
                if self._validator.is_completely_empty_row(raw_row):
                    rows_failed += 1
                    self._record_error(
                        captured_errors,
                        RowValidationError(
                            row_number=row_number,
                            column=None,
                            message="Completely empty rows are not allowed.",
                            value=None,
                        ),
                    )
                    continue

                mapped_row = self._mapper.map_row(raw_row=raw_row, mapping=mapping)
                parsed_row, row_errors = self._validator.validate_mapped_row(
                    mapped_row=mapped_row,
                    row_number=row_number,
                )
                if row_errors:
                    rows_failed += 1
                    for error in row_errors:
                        self._record_error(captured_errors, error)
                    continue
                if parsed_row is None:
                    rows_failed += 1
                    self._record_error(
                        captured_errors,
                        RowValidationError(
                            row_number=row_number,
                            column=None,
                            message="Row could not be parsed into canonical shape.",
                            value=None,
                        ),
                    )
                    continue

                batch.append(parsed_row)
                if len(batch) >= self._batch_size:
                    rows_processed += self._persist_batch(
                        repository=repository,
                        db=db,
                        batch=batch,
                    )
                    batch.clear()

            if batch:
                rows_processed += self._persist_batch(
                    repository=repository,
                    db=db,
                    batch=batch,
                )

            return IngestionSummary(
                rows_processed=rows_processed,
                rows_failed=rows_failed,
                validation_errors=captured_errors,
            )
        except UnicodeDecodeError as exc:
            raise CSVHeaderValidationError("CSV must be UTF-8 encoded.") from exc
        except csv.Error as exc:
            raise CSVHeaderValidationError(f"Invalid CSV format: {exc}") from exc
        finally:
            if text_stream is not None:
                try:
                    text_stream.detach()
                except ValueError:
                    pass

    def _persist_batch(
        self,
        *,
        repository: CanonicalInsightRepository,
        db: Session,
        batch: list[CanonicalInsightInput],
    ) -> int:
        if not batch:
            return 0

        records = [self._mapper.to_canonical_record(row) for row in batch]
        try:
            inserted = repository.bulk_insert_models(records)
            db.commit()
            return inserted
        except SQLAlchemyError as exc:
            db.rollback()
            raise CSVPersistenceError("Failed to persist valid CSV rows.") from exc

    def _resolve_mapping(
        self,
        *,
        db: Session,
        headers: list[str],
        client_name: str | None,
        mapping_config_name: str | None,
        manual_mapping: Mapping[str, str] | None,
    ) -> MappingResolution:
        mapping_repository = MappingConfigRepository(db)
        mapping_config = mapping_repository.get_active(
            name=mapping_config_name,
            client_name=client_name,
        )
        try:
            return self._mapper.resolve_mapping(
                headers=headers,
                manual_overrides=manual_mapping,
                mapping_config=mapping_config,
            )
        except SchemaMappingError as exc:
            raise CSVSchemaMappingError(message=exc.message, errors=list(exc.errors)) from exc

    def _record_error(
        self,
        captured_errors: list[RowValidationError],
        error: RowValidationError,
    ) -> None:
        if self._log_validation_errors:
            logger.warning(
                "CSV validation error row=%s column=%s message=%s value=%r",
                error.row_number,
                error.column,
                error.message,
                error.value,
            )

        if len(captured_errors) < self._max_validation_errors:
            captured_errors.append(error)


@lru_cache(maxsize=1)
def get_csv_ingestion_service() -> CSVIngestionService:
    """
    Build and cache the ingestion service with env-driven settings.
    """

    settings = get_csv_ingestion_settings()
    return CSVIngestionService(
        batch_size=settings.batch_size,
        max_validation_errors=settings.max_validation_errors,
        log_validation_errors=settings.log_validation_errors,
    )
