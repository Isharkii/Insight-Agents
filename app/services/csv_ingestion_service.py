"""
app/services/csv_ingestion_service.py

CSV ingestion with pandas chunked reading and vectorized validation.

Pipeline: read (pandas) → map (rename) → validate (vectorized) → persist (bulk).
No downstream analytics coupling.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from fastapi import UploadFile
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import get_csv_ingestion_settings
from app.domain.canonical_insight import (
    CanonicalInsightInput,
    IngestionSummary,
    RowValidationError,
)
from app.mappers.schema_mapper import MappingResolution, SchemaMapper
from app.mappers.wide_to_long_normalizer import WideToLongNormalizer
from app.repositories.canonical_insight_repository import CanonicalInsightRepository
from app.repositories.mapping_config_repository import MappingConfigRepository
from app.validators.csv_validator import (
    TIMESTAMP_REQUIRED_CATEGORIES,
    category_requires_timestamp,
    normalize_category,
    parse_timestamp_with_dateutil,
)
from app.validators.mapping_validator import MappingErrorDetail, SchemaMappingError
from db.models.canonical_insight_record import CanonicalSourceType

logger = logging.getLogger(__name__)

_REQUIRED_FIELDS: frozenset[str] = frozenset(
    {"entity_name", "category", "metric_name", "metric_value"}
)

_ALLOWED_SOURCE_TYPES: frozenset[str] = frozenset(
    {CanonicalSourceType.CSV, CanonicalSourceType.API, CanonicalSourceType.SCRAPE}
)

_ALLOWED_MULTI_ENTITY_BEHAVIORS: frozenset[str] = frozenset({"split", "error"})
_DEFAULT_MULTI_ENTITY_BEHAVIOR = "error"
_DEFAULT_ROLE = "organization"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CSVHeaderValidationError(ValueError):
    """Raised when CSV shape/header validation fails."""


class CSVPersistenceError(RuntimeError):
    """Raised when valid rows cannot be persisted."""


class CSVSchemaMappingError(CSVHeaderValidationError):
    """Raised when CSV schema mapping resolution fails with structured details."""

    def __init__(self, *, message: str, errors: list[MappingErrorDetail]) -> None:
        super().__init__(message)
        self.errors = tuple(errors)

    def to_dict(self) -> dict[str, object]:
        return {
            "message": str(self),
            "errors": [
                {
                    "code": e.code,
                    "message": e.message,
                    "canonical_field": e.canonical_field,
                    "source_column": e.source_column,
                    "context": e.context,
                }
                for e in self.errors
            ],
        }


# ---------------------------------------------------------------------------
# Metric-value coercion (numpy-backed)
# ---------------------------------------------------------------------------


def _coerce_metric_value(raw: str) -> Any:
    """Coerce a pre-validated, stripped metric_value string to Python type.

    Priority: bool → int (numpy) → float (numpy) → parsed JSON → string.
    """
    if not raw:
        return raw
    low = raw.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        fval = np.float64(raw)
        if "." not in raw and np.isfinite(fval):
            return int(fval)
        return float(fval)
    except (ValueError, OverflowError):
        pass
    if raw[0] in ("{", "["):
        return json.loads(raw)  # already validated upstream
    return raw


def _parse_metadata_cell(raw: str) -> dict[str, Any] | None:
    """Parse a pre-validated metadata_json cell. Returns None for blank."""
    if not raw:
        return None
    return json.loads(raw)  # already validated upstream


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class CSVIngestionService:
    """
    Pandas-based CSV ingestion with chunked reading and vectorized validation.

    Decoupled from analytics engines. Accepts any category value.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        max_validation_errors: int,
        log_validation_errors: bool,
        mapper: SchemaMapper | None = None,
        validator: Any = None,  # backward compat, unused
    ) -> None:
        self._batch_size = max(1, batch_size)
        self._chunksize = max(1000, batch_size)
        self._max_validation_errors = max(1, max_validation_errors)
        self._log_validation_errors = log_validation_errors
        self._mapper = mapper or SchemaMapper()

    def ingest_csv(
        self,
        *,
        upload_file: UploadFile,
        db: Session,
        client_name: str | None = None,
        mapping_config_name: str | None = None,
        manual_mapping: Mapping[str, str] | None = None,
        multi_entity_behavior: str | None = None,
        pre_detected_entities: Sequence[str] | None = None,
    ) -> IngestionSummary:
        """
        Chunked CSV ingestion: parse → map → validate → persist.

        Args:
            upload_file:          File to ingest.
            db:                   Active SQLAlchemy session (caller owns lifecycle).
            client_name:          Logical entity name for mapping config lookup.
            mapping_config_name:  Optional named mapping config to look up.
            manual_mapping:       Optional column-to-canonical overrides.
            multi_entity_behavior:
                "split" to process each entity independently, "error" to return
                a validation error when multiple entity_name values are present.
            pre_detected_entities:
                Optional entity list supplied by a caller that already scanned
                the CSV (avoids duplicate reads).

        Returns:
            IngestionSummary with row counts and any validation errors.
        """
        raw_file = upload_file.file
        raw_file.seek(0)

        try:
            # --- Peek headers ---
            try:
                header_df = pd.read_csv(raw_file, nrows=0, encoding="utf-8-sig")
            except pd.errors.EmptyDataError as exc:
                raise CSVHeaderValidationError("CSV header row is missing.") from exc

            headers = [str(c).strip() for c in header_df.columns if str(c).strip()]
            if not headers:
                raise CSVHeaderValidationError("CSV header row is missing.")

            # --- Peek sample rows for interpreter scoring ---
            raw_file.seek(0)
            sample_df = pd.read_csv(
                raw_file, nrows=20, dtype=str, encoding="utf-8-sig", keep_default_na=False,
            )

            # --- Wide-to-long detection & normalisation ---
            wide_normalizer = WideToLongNormalizer()
            wide_meta = wide_normalizer.detect(sample_df)
            wide_normalized_df: pd.DataFrame | None = None

            if len(wide_meta.wide_columns_detected) >= 2:
                raw_file.seek(0)
                full_df = pd.read_csv(
                    raw_file, dtype=str, encoding="utf-8-sig", keep_default_na=False,
                )
                wide_normalized_df, wide_meta = wide_normalizer.normalize(
                    full_df, meta=wide_meta,
                )
                headers = [
                    str(c).strip()
                    for c in wide_normalized_df.columns
                    if str(c).strip()
                ]
                sample_df = wide_normalized_df.head(20)
                logger.info(
                    "Wide-to-long normalisation applied: "
                    "preserved=%d dropped=%d periodicity=%s",
                    wide_meta.preserved_rows,
                    wide_meta.dropped_rows,
                    wide_meta.inferred_periodicity,
                )

            sample_rows: list[dict[str, str]] = [
                {str(col).strip(): str(val) for col, val in row.items()}
                for _, row in sample_df.iterrows()
            ]

            # --- Resolve canonical mapping ---
            mapping = self._resolve_mapping(
                db=db,
                headers=headers,
                client_name=client_name,
                mapping_config_name=mapping_config_name,
                manual_mapping=manual_mapping,
                sample_rows=sample_rows,
            )
            if "entity_name" not in mapping.canonical_to_source:
                return self._summary_with_error(
                    code="entity_name_unresolved",
                    message=(
                        "entity_name could not be determined from CSV mapping. "
                        "Provide a valid entity_name column or mapping override."
                    ),
                    column="entity_name",
                    context={"required_field": "entity_name"},
                )
            if "category" not in mapping.canonical_to_source:
                return self._summary_with_error(
                    code="category_missing",
                    message=(
                        "category could not be determined from CSV mapping. "
                        "Provide a valid category column or mapping override."
                    ),
                    column="category",
                    context={"required_field": "category"},
                )

            rename_map = {v: k for k, v in mapping.canonical_to_source.items()}
            usecols = list(mapping.canonical_to_source.values())
            behavior = self._resolve_multi_entity_behavior(multi_entity_behavior)
            detected_entities = sorted(
                {
                    str(entity).strip()
                    for entity in (pre_detected_entities or [])
                    if str(entity).strip()
                }
            )
            if not detected_entities:
                detected_entities = self._detect_entities_for_mapping(
                    raw_file=raw_file,
                    usecols=usecols,
                    rename_map=rename_map,
                )
            if not detected_entities:
                return self._summary_with_error(
                    code="entity_name_unresolved",
                    message=(
                        "entity_name could not be determined from CSV rows. "
                        "Ensure at least one non-empty entity_name value exists."
                    ),
                    column="entity_name",
                    context={"required_field": "entity_name"},
                )

            detected_categories = self._detect_categories_for_mapping(
                raw_file=raw_file,
                usecols=usecols,
                rename_map=rename_map,
            )
            if not detected_categories:
                return self._summary_with_error(
                    code="category_missing",
                    message=(
                        "category is missing from CSV rows. "
                        "Ensure at least one non-empty category value exists."
                    ),
                    column="category",
                    context={"required_field": "category"},
                )

            if len(detected_entities) > 1 and behavior == "error":
                return IngestionSummary(
                    rows_processed=0,
                    rows_failed=0,
                    validation_errors=[
                        RowValidationError(
                            row_number=1,
                            column="entity_name",
                            code="multiple_entities_detected",
                            message=(
                                "Multiple entity_name values detected in one CSV. "
                                "Use multi_entity_behavior='split' to allow ingest."
                            ),
                            context={
                                "entities": detected_entities,
                                "multi_entity_behavior": behavior,
                            },
                        )
                    ],
                )

            # --- Chunked reading + processing ---
            raw_file.seek(0)
            repository = CanonicalInsightRepository(db)
            rows_processed = 0
            rows_failed = 0
            captured_errors: list[RowValidationError] = []

            if wide_normalized_df is not None:
                # Wide-to-long already produced a fully normalised DataFrame;
                # iterate over it in memory-sized chunks instead of re-reading
                # the (wide-format) raw file.
                norm_usecols = [
                    c for c in usecols
                    if c in wide_normalized_df.columns
                ]
                if norm_usecols:
                    norm_df = wide_normalized_df[norm_usecols]
                else:
                    norm_df = wide_normalized_df
                chunk_iter: Any = [
                    norm_df.iloc[start:start + self._chunksize]
                    for start in range(0, len(norm_df), self._chunksize)
                ]
            else:
                chunk_iter = pd.read_csv(
                    raw_file,
                    chunksize=self._chunksize,
                    dtype=str,
                    encoding="utf-8-sig",
                    usecols=usecols,
                    keep_default_na=False,
                )

            for chunk in chunk_iter:
                chunk = chunk.rename(columns=rename_map)

                # Add missing optional columns
                for col in ("source_type", "role", "region", "metadata_json"):
                    if col not in chunk.columns:
                        if col == "source_type":
                            chunk[col] = CanonicalSourceType.CSV
                        elif col == "role":
                            chunk[col] = _DEFAULT_ROLE
                        else:
                            chunk[col] = ""
                if behavior == "split" and len(detected_entities) > 1:
                    entity_series = chunk["entity_name"].astype(str).str.strip()
                    chunk_entities = [
                        entity
                        for entity in detected_entities
                        if (entity_series == entity).any()
                    ]
                    for entity in chunk_entities:
                        entity_chunk = chunk.loc[entity_series == entity]
                        valid_inputs, n_failed = self._validate_and_normalize_chunk(
                            entity_chunk, captured_errors
                        )
                        rows_failed += n_failed
                        if valid_inputs:
                            rows_processed += self._persist_batch(
                                repository=repository, db=db, batch=valid_inputs
                            )
                    remainder_mask = ~entity_series.isin(chunk_entities)
                    if remainder_mask.any():
                        remainder_chunk = chunk.loc[remainder_mask]
                        valid_inputs, n_failed = self._validate_and_normalize_chunk(
                            remainder_chunk, captured_errors
                        )
                        rows_failed += n_failed
                        if valid_inputs:
                            rows_processed += self._persist_batch(
                                repository=repository, db=db, batch=valid_inputs
                            )
                else:
                    valid_inputs, n_failed = self._validate_and_normalize_chunk(
                        chunk, captured_errors
                    )
                    rows_failed += n_failed

                    if valid_inputs:
                            rows_processed += self._persist_batch(
                                repository=repository, db=db, batch=valid_inputs
                            )

            # Surface interpreter warnings as validation warnings
            if mapping.interpretation:
                for warn_msg in mapping.interpretation.warnings:
                    self._record_error(
                        captured_errors,
                        RowValidationError(
                            row_number=0,
                            code="schema_interpreter_warning",
                            message=warn_msg,
                        ),
                    )

            summary = IngestionSummary(
                rows_processed=rows_processed,
                rows_failed=rows_failed,
                validation_errors=captured_errors,
            )
            if summary.rows_processed == 0:
                no_valid_error = RowValidationError(
                    row_number=1,
                    code="no_valid_records",
                    message=(
                        "No valid records were ingested. "
                        "Review validation_errors for row-level details."
                    ),
                    context={
                        "rows_failed": summary.rows_failed,
                        "validation_error_count": len(summary.validation_errors),
                    },
                )
                return IngestionSummary(
                    rows_processed=0,
                    rows_failed=summary.rows_failed,
                    validation_errors=[*summary.validation_errors, no_valid_error],
                )
            return summary

        except UnicodeDecodeError as exc:
            return self._summary_with_error(
                code="csv_encoding_error",
                message="CSV must be UTF-8 encoded.",
                context={"error": str(exc)},
            )
        except pd.errors.ParserError as exc:
            return self._summary_with_error(
                code="csv_format_error",
                message=f"Invalid CSV format: {exc}",
                context={"error": str(exc)},
            )
        except CSVSchemaMappingError as exc:
            return self._summary_with_error(
                code="schema_mapping_error",
                message=str(exc),
                context={"errors": exc.to_dict().get("errors", [])},
            )
        except CSVHeaderValidationError as exc:
            return self._summary_with_error(
                code="csv_header_error",
                message=str(exc),
            )
        except CSVPersistenceError as exc:
            return self._summary_with_error(
                code="csv_persistence_error",
                message=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unhandled ingestion exception")
            return self._summary_with_error(
                code="ingestion_unexpected_error",
                message="Unexpected ingestion failure.",
                context={"error": str(exc)},
            )

    def detect_csv_entities(
        self,
        *,
        upload_file: UploadFile,
        db: Session,
        client_name: str | None = None,
        mapping_config_name: str | None = None,
        manual_mapping: Mapping[str, str] | None = None,
    ) -> list[str]:
        """Return unique entity_name values found in a CSV after schema mapping."""
        raw_file = upload_file.file
        raw_file.seek(0)

        try:
            try:
                header_df = pd.read_csv(raw_file, nrows=0, encoding="utf-8-sig")
            except pd.errors.EmptyDataError as exc:
                raise CSVHeaderValidationError("CSV header row is missing.") from exc

            headers = [str(c).strip() for c in header_df.columns if str(c).strip()]
            if not headers:
                raise CSVHeaderValidationError("CSV header row is missing.")

            mapping = self._resolve_mapping(
                db=db,
                headers=headers,
                client_name=client_name,
                mapping_config_name=mapping_config_name,
                manual_mapping=manual_mapping,
            )
            rename_map = {v: k for k, v in mapping.canonical_to_source.items()}
            usecols = list(mapping.canonical_to_source.values())
            return self._detect_entities_for_mapping(
                raw_file=raw_file,
                usecols=usecols,
                rename_map=rename_map,
            )
        except UnicodeDecodeError as exc:
            raise CSVHeaderValidationError("CSV must be UTF-8 encoded.") from exc
        except pd.errors.ParserError as exc:
            raise CSVHeaderValidationError(f"Invalid CSV format: {exc}") from exc

    # ------------------------------------------------------------------
    # Vectorized chunk validation + normalization
    # ------------------------------------------------------------------

    def _validate_and_normalize_chunk(
        self,
        chunk: pd.DataFrame,
        captured_errors: list[RowValidationError],
    ) -> tuple[list[CanonicalInsightInput], int]:
        """Validate and normalize one DataFrame chunk.

        Returns (valid_inputs, n_failed). Appends errors to captured_errors.
        """
        if chunk.empty:
            return [], 0

        # Vectorized strip on all cells
        stripped = chunk.apply(lambda col: col.str.strip())
        invalid = pd.Series(False, index=chunk.index)

        # Phase 1: Empty rows
        empty_mask = (stripped == "").all(axis=1)
        invalid |= empty_mask
        for idx in chunk.index[empty_mask]:
            self._record_error(
                captured_errors,
                RowValidationError(
                    row_number=int(idx) + 2,
                    code="empty_row",
                    message="Completely empty rows are not allowed.",
                ),
            )

        # Phase 2: Required fields (vectorized set comparison)
        active = ~invalid
        for col in _REQUIRED_FIELDS:
            if col in stripped.columns:
                missing = active & (stripped[col] == "")
            else:
                missing = active.copy()
            if missing.any():
                invalid |= missing
                active = ~invalid
                for idx in chunk.index[missing]:
                    self._record_error(
                        captured_errors,
                        RowValidationError(
                            row_number=int(idx) + 2,
                            column=col,
                            code="required_value_missing",
                            message="Required value is missing.",
                            value=chunk.at[idx, col] if col in chunk.columns else None,
                            context={"field": col},
                        ),
                    )

        # Phase 3: source_type normalization — default invalid values to "csv"
        if "source_type" in stripped.columns:
            st_lower = stripped["source_type"].str.lower()
            bad_source = ~st_lower.isin(_ALLOWED_SOURCE_TYPES)
            if bad_source.any():
                stripped.loc[bad_source, "source_type"] = CanonicalSourceType.CSV

        # Phase 4: Timestamp policy + parsing.
        # Required for specific categories; optional for static/other datasets.
        active = ~invalid
        ts_parsed = pd.Series(pd.NaT, index=chunk.index)
        ts_raw = (
            stripped["timestamp"]
            if "timestamp" in stripped.columns
            else pd.Series("", index=chunk.index, dtype="object")
        )
        category_series = stripped["category"].apply(normalize_category)

        requires_ts = active & category_series.apply(category_requires_timestamp)
        missing_required_ts = requires_ts & (ts_raw == "")
        if missing_required_ts.any():
            invalid |= missing_required_ts
            for idx in chunk.index[missing_required_ts]:
                self._record_error(
                    captured_errors,
                    RowValidationError(
                        row_number=int(idx) + 2,
                        column="timestamp",
                        code="timestamp_required_for_category",
                        message="timestamp is required for this category.",
                        value=chunk.at[idx, "timestamp"] if "timestamp" in chunk.columns else None,
                        context={
                            "category": category_series.at[idx],
                            "required_categories": sorted(TIMESTAMP_REQUIRED_CATEGORIES),
                        },
                    ),
                )

        active = ~invalid
        parse_mask = active & (ts_raw != "")
        parse_failures = pd.Series(False, index=chunk.index)
        for idx in chunk.index[parse_mask]:
            raw_value = ts_raw.at[idx]
            try:
                parsed = parse_timestamp_with_dateutil(raw_value)
                ts_parsed.at[idx] = pd.Timestamp(parsed)
            except Exception:
                parse_failures.at[idx] = True
                self._record_error(
                    captured_errors,
                    RowValidationError(
                        row_number=int(idx) + 2,
                        column="timestamp",
                        code="timestamp_invalid_format",
                        message="Invalid date/time format.",
                        value=chunk.at[idx, "timestamp"] if "timestamp" in chunk.columns else None,
                        context={"category": category_series.at[idx]},
                    ),
                )
        if parse_failures.any():
            invalid |= parse_failures

        # Phase 5: metric_value JSON validation (per-cell for JSON-looking only)
        active = ~invalid
        if "metric_value" in stripped.columns:
            mv = stripped["metric_value"]
            looks_json = active & (mv.str.startswith("{") | mv.str.startswith("["))
            if looks_json.any():
                json_bad = pd.Series(False, index=chunk.index)
                for idx in chunk.index[looks_json]:
                    try:
                        json.loads(mv.at[idx])
                    except (json.JSONDecodeError, ValueError):
                        json_bad.at[idx] = True
                        self._record_error(
                            captured_errors,
                            RowValidationError(
                                row_number=int(idx) + 2,
                                column="metric_value",
                                code="metric_value_json_invalid",
                                message="metric_value JSON could not be parsed.",
                                value=chunk.at[idx, "metric_value"],
                            ),
                        )
                invalid |= json_bad

        # Phase 6: metadata_json validation (per-cell for non-blank only)
        active = ~invalid
        if "metadata_json" in stripped.columns:
            md = stripped["metadata_json"]
            has_md = active & (md != "")
            if has_md.any():
                md_bad = pd.Series(False, index=chunk.index)
                for idx in chunk.index[has_md]:
                    try:
                        parsed = json.loads(md.at[idx])
                        if not isinstance(parsed, dict):
                            md_bad.at[idx] = True
                            self._record_error(
                                captured_errors,
                                RowValidationError(
                                    row_number=int(idx) + 2,
                                    column="metadata_json",
                                    code="metadata_json_not_object",
                                    message="metadata_json must be a JSON object.",
                                    value=chunk.at[idx, "metadata_json"],
                                ),
                            )
                    except json.JSONDecodeError:
                        md_bad.at[idx] = True
                        self._record_error(
                            captured_errors,
                            RowValidationError(
                                row_number=int(idx) + 2,
                                column="metadata_json",
                                code="metadata_json_invalid_json",
                                message="metadata_json must be valid JSON object.",
                                value=chunk.at[idx, "metadata_json"],
                            ),
                        )
                invalid |= md_bad

        # --- Build valid records ---
        n_failed = int(invalid.sum())
        valid_idx = chunk.index[~invalid]
        if valid_idx.empty:
            return [], n_failed

        return self._build_inputs(stripped, ts_parsed, valid_idx), n_failed

    # ------------------------------------------------------------------
    # Record construction from validated chunk
    # ------------------------------------------------------------------

    @staticmethod
    def _build_inputs(
        stripped: pd.DataFrame,
        ts_parsed: pd.Series,
        valid_idx: pd.Index,
    ) -> list[CanonicalInsightInput]:
        """Construct CanonicalInsightInput list from validated rows."""
        s = stripped.loc[valid_idx]

        # Vectorized normalization
        source_type = s["source_type"].str.lower()
        entity_name = s["entity_name"]
        category = s["category"].str.lower()
        metric_name = s["metric_name"]

        # Metric value: numpy-backed coercion via apply
        metric_value = s["metric_value"].apply(_coerce_metric_value)

        # Timestamps: already parsed, convert to Python datetime
        timestamps = ts_parsed.loc[valid_idx]
        fallback_timestamp = datetime.now(tz=timezone.utc)

        # Optional columns
        region = (
            s["region"].where(s["region"] != "", None)
            if "region" in s.columns
            else pd.Series(None, index=valid_idx)
        )
        role = (
            s["role"].where(s["role"] != "", _DEFAULT_ROLE)
            if "role" in s.columns
            else pd.Series(_DEFAULT_ROLE, index=valid_idx)
        )
        metadata_json = (
            s["metadata_json"].apply(_parse_metadata_cell)
            if "metadata_json" in s.columns
            else pd.Series(None, index=valid_idx)
        )

        inputs: list[CanonicalInsightInput] = []
        for idx in valid_idx:
            ts = timestamps.at[idx]
            if pd.isna(ts):
                ts = fallback_timestamp
            elif hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()

            rgn = region.at[idx]
            if isinstance(rgn, float) and np.isnan(rgn):
                rgn = None

            role_value = role.at[idx]
            if isinstance(role_value, float) and np.isnan(role_value):
                role_value = _DEFAULT_ROLE
            role_str = str(role_value).strip() if role_value is not None else ""
            if not role_str:
                role_str = _DEFAULT_ROLE

            md = metadata_json.at[idx]
            if isinstance(md, float) and not isinstance(md, bool):
                md = None

            inputs.append(
                CanonicalInsightInput(
                    source_type=source_type.at[idx],
                    entity_name=entity_name.at[idx],
                    category=category.at[idx],
                    role=role_str,
                    metric_name=metric_name.at[idx],
                    metric_value=metric_value.at[idx],
                    timestamp=ts,
                    region=rgn,
                    metadata_json=md,
                )
            )
        return inputs

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_batch(
        self,
        *,
        repository: CanonicalInsightRepository,
        db: Session,
        batch: list[CanonicalInsightInput],
    ) -> int:
        if not batch:
            return 0
        try:
            inserted = repository.bulk_insert(batch, batch_size=self._batch_size)
            db.commit()
            return inserted
        except SQLAlchemyError as exc:
            db.rollback()
            raise CSVPersistenceError("Failed to persist valid CSV rows.") from exc

    # ------------------------------------------------------------------
    # Mapping resolution
    # ------------------------------------------------------------------

    def _resolve_mapping(
        self,
        *,
        db: Session,
        headers: list[str],
        client_name: str | None,
        mapping_config_name: str | None,
        manual_mapping: Mapping[str, str] | None,
        sample_rows: Sequence[Mapping[str, str]] | None = None,
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
                sample_rows=sample_rows,
            )
        except SchemaMappingError as exc:
            raise CSVSchemaMappingError(
                message=exc.message, errors=list(exc.errors)
            ) from exc

    def _detect_entities_for_mapping(
        self,
        *,
        raw_file: Any,
        usecols: Sequence[str],
        rename_map: Mapping[str, str],
    ) -> list[str]:
        """Scan CSV and return sorted unique, non-empty entity_name values."""
        return self._detect_distinct_values_for_mapping(
            raw_file=raw_file,
            usecols=usecols,
            rename_map=rename_map,
            canonical_field="entity_name",
        )

    def _detect_categories_for_mapping(
        self,
        *,
        raw_file: Any,
        usecols: Sequence[str],
        rename_map: Mapping[str, str],
    ) -> list[str]:
        """Scan CSV and return sorted unique, non-empty category values."""
        return self._detect_distinct_values_for_mapping(
            raw_file=raw_file,
            usecols=usecols,
            rename_map=rename_map,
            canonical_field="category",
        )

    def _detect_distinct_values_for_mapping(
        self,
        *,
        raw_file: Any,
        usecols: Sequence[str],
        rename_map: Mapping[str, str],
        canonical_field: str,
    ) -> list[str]:
        """Scan CSV and return sorted unique non-empty values for one field."""
        raw_file.seek(0)
        values_set: set[str] = set()
        reader = pd.read_csv(
            raw_file,
            chunksize=self._chunksize,
            dtype=str,
            encoding="utf-8-sig",
            usecols=list(usecols),
            keep_default_na=False,
        )
        for chunk in reader:
            chunk = chunk.rename(columns=rename_map)
            if canonical_field not in chunk.columns:
                continue
            values = chunk[canonical_field].astype(str).str.strip()
            values_set.update(value for value in values.tolist() if value)
        raw_file.seek(0)
        return sorted(values_set)

    def _resolve_multi_entity_behavior(self, multi_entity_behavior: str | None) -> str:
        raw = (
            multi_entity_behavior
            or os.getenv("CSV_MULTI_ENTITY_BEHAVIOR")
            or _DEFAULT_MULTI_ENTITY_BEHAVIOR
        )
        normalized = str(raw).strip().lower()
        if normalized not in _ALLOWED_MULTI_ENTITY_BEHAVIORS:
            allowed = ", ".join(sorted(_ALLOWED_MULTI_ENTITY_BEHAVIORS))
            raise CSVHeaderValidationError(
                f"Invalid multi_entity_behavior '{raw}'. Allowed values: {allowed}."
            )
        return normalized

    @staticmethod
    def _summary_with_error(
        *,
        code: str,
        message: str,
        row_number: int = 1,
        column: str | None = None,
        value: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> IngestionSummary:
        return IngestionSummary(
            rows_processed=0,
            rows_failed=0,
            validation_errors=[
                RowValidationError(
                    row_number=row_number,
                    column=column,
                    code=code,
                    message=message,
                    value=value,
                    context=context,
                )
            ],
        )

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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_csv_ingestion_service() -> CSVIngestionService:
    """Build and cache the ingestion service with env-driven settings."""
    settings = get_csv_ingestion_settings()
    return CSVIngestionService(
        batch_size=settings.batch_size,
        max_validation_errors=settings.max_validation_errors,
        log_validation_errors=settings.log_validation_errors,
    )
