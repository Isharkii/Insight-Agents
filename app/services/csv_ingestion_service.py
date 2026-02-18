"""
app/services/csv_ingestion_service.py

Service layer for CSV ingestion workflow orchestration.

After a successful ingestion (rows_processed > 0), the service triggers
the downstream analytics pipeline in this order:

    1. KPIOrchestrator.run()                       — recomputes KPIs (90-day window)
    2. ForecastOrchestrator.generate_forecast()    — generates metric forecast
    3. RiskOrchestrator.generate_risk_score()      — scores business risk
    4. SegmentationOrchestrator.run_segmentation() — clusters entity data

Each step is independent: a failure is logged at WARNING level and does not
affect subsequent steps or the ingestion result. KPIOrchestrator commits
internally; Forecast / Risk / Segmentation are committed explicitly here.
"""

from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, timedelta, timezone
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
from app.services.kpi_orchestrator import KPIOrchestrator, KPIRunResult
from app.validators.mapping_validator import MappingErrorDetail, SchemaMappingError
from app.validators.csv_validator import CSVRowValidator
from forecast.orchestrator import ForecastOrchestrator
from risk.orchestrator import RiskOrchestrator
from segmentation.orchestrator import SegmentationOrchestrator

logger = logging.getLogger(__name__)

# Primary metric to seed the forecast pipeline per business type.
# ForecastOrchestrator returns {"error": "Insufficient data."} when fewer
# than 2 values are supplied — expected on first ingestion; not an error.
_PRIMARY_METRIC_BY_BUSINESS_TYPE: dict[str, str] = {
    "saas": "mrr",
    "ecommerce": "revenue",
    "agency": "total_revenue",
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


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
        business_type: str | None = None,
    ) -> IngestionSummary:
        """
        Stream a CSV file, skip invalid rows, and persist valid rows in batches.

        When ``client_name`` and ``business_type`` are both provided and at
        least one row is persisted, the downstream analytics pipeline is
        triggered automatically (KPI → Forecast → Risk → Segmentation).
        Analytics failures do not affect the returned IngestionSummary.

        Args:
            upload_file:          File to ingest.
            db:                   Active SQLAlchemy session (caller owns lifecycle).
            client_name:          Logical entity name; doubles as entity_name for
                                  analytics orchestrators.
            mapping_config_name:  Optional named mapping config to look up.
            manual_mapping:       Optional column-to-canonical overrides.
            business_type:        One of "saas", "ecommerce", "agency". Required
                                  to trigger the analytics pipeline.
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

            summary = IngestionSummary(
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

        # Trigger analytics pipeline once ingestion is confirmed complete.
        if rows_processed > 0 and client_name and business_type:
            self._trigger_analytics_pipeline(
                entity_name=client_name,
                business_type=business_type,
                db=db,
            )

        return summary

    # ------------------------------------------------------------------
    # Post-ingestion analytics trigger
    # ------------------------------------------------------------------

    def _trigger_analytics_pipeline(
        self,
        *,
        entity_name: str,
        business_type: str,
        db: Session,
    ) -> None:
        """Fire downstream orchestrators after a successful ingestion.

        Each step runs inside its own try/except so a failure in one step
        does not prevent subsequent steps from running. Failures are logged
        at WARNING level and never propagate to the caller.

        Transaction contract:
          - KPIOrchestrator commits internally (inside _persist).
          - ForecastOrchestrator, RiskOrchestrator, SegmentationOrchestrator
            do NOT commit — explicit commit / rollback is applied here.
        """
        now = datetime.now(tz=timezone.utc)
        period_start = now - timedelta(days=90)

        kpi_result: KPIRunResult | None = None
        forecast_data_for_risk: dict = {}

        # --- Step 1: KPI recomputation ---
        try:
            kpi_result = KPIOrchestrator().run(
                entity_name=entity_name,
                business_type=business_type,
                period_start=period_start,
                period_end=now,
                db=db,
            )
            logger.info(
                "Post-ingestion KPI computed entity=%r business_type=%r "
                "record_id=%s has_errors=%s",
                entity_name,
                business_type,
                kpi_result.record_id,
                kpi_result.has_errors,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Post-ingestion KPI computation failed entity=%r business_type=%r: %s",
                entity_name,
                business_type,
                exc,
            )

        # --- Step 2: Forecast generation ---
        # Pass a single-element series derived from the KPI result.
        # ForecastOrchestrator returns {"error": "Insufficient data."} when
        # fewer than 2 values are supplied — this is expected on first ingestion
        # and does not trigger a DB write, so no commit / rollback is needed.
        try:
            metric_name = _PRIMARY_METRIC_BY_BUSINESS_TYPE.get(
                business_type, "revenue"
            )
            values = _extract_primary_metric_values(kpi_result, metric_name)
            forecast_result = ForecastOrchestrator(db).generate_forecast(
                entity_name=entity_name,
                metric_name=metric_name,
                values=values,
            )
            if "error" in forecast_result:
                logger.info(
                    "Post-ingestion forecast deferred entity=%r metric=%r: %s",
                    entity_name,
                    metric_name,
                    forecast_result["error"],
                )
            else:
                db.commit()
                forecast_data_for_risk = forecast_result
                logger.info(
                    "Post-ingestion forecast generated entity=%r metric=%r trend=%s",
                    entity_name,
                    metric_name,
                    forecast_result.get("trend"),
                )
        except Exception as exc:  # noqa: BLE001
            db.rollback()
            logger.warning(
                "Post-ingestion forecast generation failed entity=%r: %s",
                entity_name,
                exc,
            )

        # --- Step 3: Risk scoring ---
        # kpi_data signals (deltas) are not directly derivable from one ingestion
        # batch; pass empty dict so the orchestrator applies its 0.0 defaults.
        try:
            risk_result = RiskOrchestrator(db).generate_risk_score(
                entity_name=entity_name,
                kpi_data={},
                forecast_data=forecast_data_for_risk,
            )
            db.commit()
            logger.info(
                "Post-ingestion risk scored entity=%r score=%s level=%s",
                entity_name,
                risk_result.get("risk_score"),
                risk_result.get("risk_level"),
            )
        except Exception as exc:  # noqa: BLE001
            db.rollback()
            logger.warning(
                "Post-ingestion risk scoring failed entity=%r: %s",
                entity_name,
                exc,
            )

        # --- Step 4: Segmentation ---
        # Build a single-record list from the KPI metrics as a best-effort seed.
        # n_clusters must be <= n_records; skip when records are empty.
        try:
            seg_records = _build_segmentation_records(kpi_result)
            n_clusters = min(3, len(seg_records))
            if n_clusters < 1:
                logger.info(
                    "Post-ingestion segmentation skipped entity=%r: "
                    "insufficient records for clustering",
                    entity_name,
                )
            else:
                seg_result = SegmentationOrchestrator(session=db).run_segmentation(
                    entity_name=entity_name,
                    records=seg_records,
                    n_clusters=n_clusters,
                )
                db.commit()
                logger.info(
                    "Post-ingestion segmentation completed entity=%r n_clusters=%s",
                    entity_name,
                    seg_result.get("n_clusters"),
                )
        except Exception as exc:  # noqa: BLE001
            db.rollback()
            logger.warning(
                "Post-ingestion segmentation failed entity=%r: %s",
                entity_name,
                exc,
            )

    # ------------------------------------------------------------------
    # Ingestion internals
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


# ---------------------------------------------------------------------------
# Module-level helpers for analytics trigger (no business logic)
# ---------------------------------------------------------------------------


def _extract_primary_metric_values(
    kpi_result: KPIRunResult | None,
    metric_name: str,
) -> list[float]:
    """Return a single-element list from the named KPI metric value.

    ForecastOrchestrator needs at least 2 values for meaningful regression.
    Returning one value (or an empty list on first ingestion) causes the
    orchestrator to respond with an "Insufficient data." error dict —
    the correct signal to defer forecasting until more history exists.

    Args:
        kpi_result:  Result from KPIOrchestrator.run(), or None if KPI failed.
        metric_name: Metric key to extract from the KPI payload.

    Returns:
        A list of zero or one floats.
    """
    if kpi_result is None:
        return []
    entry = kpi_result.metrics.get(metric_name, {})
    value = entry.get("value")
    if not isinstance(value, (int, float)):
        return []
    return [float(value)]


def _build_segmentation_records(
    kpi_result: KPIRunResult | None,
) -> list[dict]:
    """Build a flat metric record list from a KPI result for segmentation.

    Extracts numeric metric values into a single dict and wraps it in a list.
    The SegmentationOrchestrator's FeatureEngineer accepts this shape.

    Args:
        kpi_result: Result from KPIOrchestrator.run(), or None if KPI failed.

    Returns:
        A list of zero or one metric dicts.
    """
    if kpi_result is None:
        return []
    flat: dict = {}
    for metric_name, entry in kpi_result.metrics.items():
        value = entry.get("value")
        if isinstance(value, (int, float)):
            flat[metric_name] = float(value)
    return [flat] if flat else []


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


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
