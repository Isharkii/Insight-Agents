"""
app/validators/csv_validator.py

Row-level validation and type parsing for CSV ingestion.

Category is accepted as any non-empty string — no hardcoded allowlist.
Source type is still validated (csv, api, scrape are structural constants).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from dateutil import parser as date_parser
from dateutil.parser import ParserError

from app.domain.canonical_insight import CanonicalInsightInput, RowValidationError
from db.models.canonical_insight_record import CanonicalSourceType

ALLOWED_SOURCE_TYPES = {
    CanonicalSourceType.CSV,
    CanonicalSourceType.API,
    CanonicalSourceType.SCRAPE,
}

TIMESTAMP_REQUIRED_CATEGORIES: frozenset[str] = frozenset(
    {"saas", "financial_market", "generic_timeseries"}
)


def normalize_category(value: str | None) -> str:
    return str(value or "").strip().lower()


def category_requires_timestamp(category: str | None) -> bool:
    return normalize_category(category) in TIMESTAMP_REQUIRED_CATEGORIES


def parse_timestamp_with_dateutil(value: str) -> datetime:
    parsed = date_parser.parse(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


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

        source_type_raw = self._parse_optional_string(mapped_row.get("source_type")) or CanonicalSourceType.CSV
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
        role = self._parse_optional_string(mapped_row.get("role")) or "organization"

        metric_value = self._parse_metric_value(
            value=mapped_row.get("metric_value"),
            row_number=row_number,
            errors=errors,
        )
        # Category: normalize to lowercase, no allowlist check.
        category = normalize_category(category_raw)
        timestamp = self._parse_timestamp(
            value=mapped_row.get("timestamp"),
            category=category,
            row_number=row_number,
            errors=errors,
        )

        source_type = self._validate_source_type(
            source_type=source_type_raw,
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
                role=role,
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
            return CanonicalSourceType.CSV

        normalized = source_type.strip().lower()
        if normalized not in ALLOWED_SOURCE_TYPES:
            return CanonicalSourceType.CSV
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
                    code="required_value_missing",
                    message="Required value is missing.",
                    value=self._stringify_value(value),
                    context={"field": column},
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
                    code="required_value_missing",
                    message="Required value is missing.",
                    value=self._stringify_value(value),
                    context={"field": "metric_value"},
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
                        code="metric_value_json_invalid",
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
        category: str | None,
        row_number: int,
        errors: list[RowValidationError],
    ) -> datetime:
        if self._is_blank(value) and category_requires_timestamp(category):
            errors.append(
                RowValidationError(
                    row_number=row_number,
                    column="timestamp",
                    code="timestamp_required_for_category",
                    message="timestamp is required for this category.",
                    value=self._stringify_value(value),
                    context={
                        "category": normalize_category(category),
                        "required_categories": sorted(TIMESTAMP_REQUIRED_CATEGORIES),
                    },
                )
            )
            return datetime.min
        if self._is_blank(value):
            return datetime.now(tz=timezone.utc)

        raw = str(value).strip()
        try:
            return parse_timestamp_with_dateutil(raw)
        except (ParserError, OverflowError, TypeError, ValueError):
            errors.append(
                RowValidationError(
                    row_number=row_number,
                    column="timestamp",
                    code="timestamp_invalid_format",
                    message="Invalid date/time format.",
                    value=self._stringify_value(raw),
                    context={"category": normalize_category(category)},
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
                    code="metadata_json_invalid_json",
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
                    code="metadata_json_not_object",
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
