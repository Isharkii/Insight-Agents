"""
app/validators/csv_validator.py

Row-level validation and type parsing for CSV ingestion.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from app.domain.canonical_insight import CanonicalInsightInput, RowValidationError
from db.models.canonical_insight_record import CanonicalCategory, CanonicalSourceType

TIMESTAMP_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
)

ALLOWED_SOURCE_TYPES = {
    CanonicalSourceType.CSV,
    CanonicalSourceType.API,
    CanonicalSourceType.SCRAPE,
}

ALLOWED_CATEGORIES = {
    CanonicalCategory.SALES,
    CanonicalCategory.MARKETING,
    CanonicalCategory.PRICING,
    CanonicalCategory.EVENT,
    CanonicalCategory.MACRO,
}


class CSVRowValidator:
    """
    Validates and parses mapped canonical row values.
    """

    def is_completely_empty_row(self, row: Mapping[str, Any]) -> bool:
        """
        Return True when all values in the row are empty or whitespace.
        """

        return all(self._is_blank(value) for value in row.values())

    def validate_mapped_row(
        self,
        *,
        mapped_row: Mapping[str, str | None],
        row_number: int,
    ) -> tuple[CanonicalInsightInput | None, list[RowValidationError]]:
        """
        Validate and parse one canonical mapped row.
        """

        errors: list[RowValidationError] = []

        source_type_raw = self._parse_required_string(
            value=mapped_row.get("source_type"),
            row_number=row_number,
            column="source_type",
            errors=errors,
        )
        entity_name = self._parse_required_string(
            value=mapped_row.get("entity_name"),
            row_number=row_number,
            column="entity_name",
            errors=errors,
        )
        category_raw = self._parse_required_string(
            value=mapped_row.get("category"),
            row_number=row_number,
            column="category",
            errors=errors,
        )
        metric_name = self._parse_required_string(
            value=mapped_row.get("metric_name"),
            row_number=row_number,
            column="metric_name",
            errors=errors,
        )

        metric_value = self._parse_metric_value(
            value=mapped_row.get("metric_value"),
            row_number=row_number,
            errors=errors,
        )
        timestamp = self._parse_timestamp(
            value=mapped_row.get("timestamp"),
            row_number=row_number,
            errors=errors,
        )

        source_type = self._validate_source_type(
            source_type=source_type_raw,
            row_number=row_number,
            errors=errors,
        )
        category = self._validate_category(
            category=category_raw,
            row_number=row_number,
            errors=errors,
        )

        region = self._parse_optional_string(mapped_row.get("region"))
        metadata_json = self._parse_optional_metadata_json(
            value=mapped_row.get("metadata_json"),
            row_number=row_number,
            errors=errors,
        )

        if errors:
            return None, errors

        return (
            CanonicalInsightInput(
                source_type=source_type,
                entity_name=entity_name,
                category=category,
                metric_name=metric_name,
                metric_value=metric_value,
                timestamp=timestamp,
                region=region,
                metadata_json=metadata_json,
            ),
            [],
        )

    def _validate_source_type(
        self,
        *,
        source_type: str | None,
        row_number: int,
        errors: list[RowValidationError],
    ) -> str:
        if source_type is None:
            return ""

        normalized = source_type.strip().lower()
        if normalized not in ALLOWED_SOURCE_TYPES:
            allowed = ", ".join(sorted(ALLOWED_SOURCE_TYPES))
            errors.append(
                RowValidationError(
                    row_number=row_number,
                    column="source_type",
                    message=f"Unsupported source_type. Allowed values: {allowed}.",
                    value=self._stringify_value(source_type),
                )
            )
        return normalized

    def _validate_category(
        self,
        *,
        category: str | None,
        row_number: int,
        errors: list[RowValidationError],
    ) -> str:
        if category is None:
            return ""

        normalized = category.strip().lower()
        if normalized not in ALLOWED_CATEGORIES:
            allowed = ", ".join(sorted(ALLOWED_CATEGORIES))
            errors.append(
                RowValidationError(
                    row_number=row_number,
                    column="category",
                    message=f"Unsupported category. Allowed values: {allowed}.",
                    value=self._stringify_value(category),
                )
            )
        return normalized

    def _parse_required_string(
        self,
        *,
        value: Any,
        row_number: int,
        column: str,
        errors: list[RowValidationError],
    ) -> str:
        if self._is_blank(value):
            errors.append(
                RowValidationError(
                    row_number=row_number,
                    column=column,
                    message="Required value is missing.",
                    value=self._stringify_value(value),
                )
            )
            return ""
        return str(value).strip()

    def _parse_optional_string(self, value: str | None) -> str | None:
        if self._is_blank(value):
            return None
        return str(value).strip()

    def _parse_metric_value(
        self,
        *,
        value: str | None,
        row_number: int,
        errors: list[RowValidationError],
    ) -> Any:
        if self._is_blank(value):
            errors.append(
                RowValidationError(
                    row_number=row_number,
                    column="metric_value",
                    message="Required value is missing.",
                    value=self._stringify_value(value),
                )
            )
            return None

        raw_value = str(value).strip()
        lowered = raw_value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False

        try:
            return int(raw_value)
        except ValueError:
            pass

        try:
            decimal_value = Decimal(raw_value)
            return float(decimal_value)
        except (InvalidOperation, ValueError):
            pass

        if raw_value.startswith("{") or raw_value.startswith("["):
            try:
                return json.loads(raw_value)
            except json.JSONDecodeError:
                errors.append(
                    RowValidationError(
                        row_number=row_number,
                        column="metric_value",
                        message="metric_value JSON could not be parsed.",
                        value=self._stringify_value(raw_value),
                    )
                )
                return None

        return raw_value

    def _parse_timestamp(
        self,
        *,
        value: str | None,
        row_number: int,
        errors: list[RowValidationError],
    ) -> datetime:
        if self._is_blank(value):
            errors.append(
                RowValidationError(
                    row_number=row_number,
                    column="timestamp",
                    message="Required value is missing.",
                    value=self._stringify_value(value),
                )
            )
            return datetime.min

        raw = str(value).strip()

        normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
        try:
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except ValueError:
            pass

        for fmt in TIMESTAMP_FORMATS:
            try:
                parsed = datetime.strptime(raw, fmt)
                return parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

        errors.append(
            RowValidationError(
                row_number=row_number,
                column="timestamp",
                message="Invalid date/time format.",
                value=self._stringify_value(raw),
            )
        )
        return datetime.min

    def _parse_optional_metadata_json(
        self,
        *,
        value: str | None,
        row_number: int,
        errors: list[RowValidationError],
    ) -> dict[str, Any] | None:
        if self._is_blank(value):
            return None

        raw = str(value).strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            errors.append(
                RowValidationError(
                    row_number=row_number,
                    column="metadata_json",
                    message="metadata_json must be valid JSON object.",
                    value=self._stringify_value(raw),
                )
            )
            return None

        if not isinstance(parsed, dict):
            errors.append(
                RowValidationError(
                    row_number=row_number,
                    column="metadata_json",
                    message="metadata_json must be a JSON object.",
                    value=self._stringify_value(raw),
                )
            )
            return None

        return parsed

    @staticmethod
    def _is_blank(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, (list, tuple)):
            return all(str(item).strip() == "" for item in value)
        return str(value).strip() == ""

    @staticmethod
    def _stringify_value(value: Any) -> str | None:
        if value is None:
            return None
        return str(value)
