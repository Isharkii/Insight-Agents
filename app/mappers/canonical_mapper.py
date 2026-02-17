"""
app/mappers/canonical_mapper.py

Column mapping from arbitrary CSV headers to canonical fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

CANONICAL_FIELDS: tuple[str, ...] = (
    "source_type",
    "entity_name",
    "category",
    "metric_name",
    "metric_value",
    "timestamp",
    "region",
    "metadata_json",
)

REQUIRED_CANONICAL_FIELDS: tuple[str, ...] = (
    "source_type",
    "entity_name",
    "category",
    "metric_name",
    "metric_value",
    "timestamp",
)

DEFAULT_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "source_type": ("source", "source kind", "source_kind", "input_source"),
    "entity_name": ("entity", "company", "client", "competitor", "account_name"),
    "category": ("insight_category", "metric_category", "domain", "topic"),
    "metric_name": ("metric", "kpi", "measure_name", "metric_key"),
    "metric_value": ("value", "metric_amount", "measure_value", "kpi_value"),
    "timestamp": ("date", "datetime", "event_time", "measured_at", "recorded_at"),
    "region": ("geo", "geography", "country", "market_region", "location"),
    "metadata_json": ("metadata", "meta", "context_json", "attributes"),
}


def normalize_header(header: str) -> str:
    """
    Normalize a column name for flexible matching.
    """

    return "".join(ch for ch in header.strip().lower() if ch.isalnum())


class MissingRequiredColumnsError(ValueError):
    """
    Raised when required canonical fields cannot be mapped from CSV headers.
    """

    def __init__(self, missing: Sequence[str], headers: Sequence[str]) -> None:
        missing_csv = ", ".join(missing)
        headers_csv = ", ".join(headers) if headers else "<none>"
        message = (
            f"Missing required columns after mapping: {missing_csv}. "
            f"CSV headers: {headers_csv}"
        )
        super().__init__(message)
        self.missing = tuple(missing)
        self.headers = tuple(headers)


@dataclass(frozen=True)
class ColumnMapping:
    """
    Resolved mapping between canonical field names and source CSV headers.
    """

    canonical_to_source: dict[str, str]
    source_headers: tuple[str, ...]


class ColumnMapper:
    """
    Maps incoming CSV columns to canonical schema fields.
    """

    def __init__(self, aliases: Mapping[str, Sequence[str]] | None = None) -> None:
        alias_map = aliases or DEFAULT_COLUMN_ALIASES
        self._aliases: dict[str, tuple[str, ...]] = {
            canonical: tuple(values)
            for canonical, values in alias_map.items()
        }

    def build_mapping(
        self,
        headers: Sequence[str],
        *,
        required_fields: Sequence[str] = REQUIRED_CANONICAL_FIELDS,
    ) -> ColumnMapping:
        """
        Resolve canonical field to source-header mapping for one CSV.
        """

        normalized_header_lookup: dict[str, str] = {}
        for header in headers:
            normalized = normalize_header(header)
            if normalized and normalized not in normalized_header_lookup:
                normalized_header_lookup[normalized] = header

        mapping: dict[str, str] = {}
        for canonical_field in CANONICAL_FIELDS:
            candidates = (canonical_field, *self._aliases.get(canonical_field, ()))
            for candidate in candidates:
                match = normalized_header_lookup.get(normalize_header(candidate))
                if match:
                    mapping[canonical_field] = match
                    break

        missing = [field for field in required_fields if field not in mapping]
        if missing:
            raise MissingRequiredColumnsError(missing=missing, headers=headers)

        return ColumnMapping(
            canonical_to_source=mapping,
            source_headers=tuple(headers),
        )

    def map_row(
        self,
        *,
        raw_row: Mapping[str, str | None],
        mapping: ColumnMapping,
    ) -> dict[str, str | None]:
        """
        Convert a source CSV row into canonical raw fields.
        """

        mapped: dict[str, str | None] = {}
        for canonical_field, source_column in mapping.canonical_to_source.items():
            mapped[canonical_field] = raw_row.get(source_column)
        return mapped
