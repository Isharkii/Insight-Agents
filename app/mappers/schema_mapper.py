"""
app/mappers/schema_mapper.py

Dynamic schema mapping engine for CSV-to-canonical mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Mapping, Sequence

from app.domain.canonical_insight import CanonicalInsightInput
from app.validators.mapping_validator import MappingErrorDetail, MappingValidator, SchemaMappingError
from db.models.canonical_insight_record import CanonicalInsightRecord

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


@dataclass(frozen=True)
class MappingResolution:
    """
    Final resolved mapping metadata.
    """

    canonical_to_source: dict[str, str]
    source_headers: tuple[str, ...]
    match_strategies: dict[str, str]
    mapping_config_id: str | None = None


class SchemaMapper:
    """
    Resolves source CSV schemas into canonical field mappings.
    """

    def __init__(
        self,
        *,
        aliases: Mapping[str, Sequence[str]] | None = None,
        validator: MappingValidator | None = None,
        fuzzy_threshold: float = 0.84,
    ) -> None:
        self._aliases: dict[str, tuple[str, ...]] = {
            canonical: tuple(values)
            for canonical, values in (aliases or DEFAULT_COLUMN_ALIASES).items()
        }
        self._validator = validator or MappingValidator(
            required_fields=REQUIRED_CANONICAL_FIELDS,
            canonical_fields=CANONICAL_FIELDS,
        )
        self._fuzzy_threshold = max(0.0, min(1.0, fuzzy_threshold))

    def resolve_mapping(
        self,
        headers: Sequence[str],
        *,
        manual_overrides: Mapping[str, str] | None = None,
        mapping_config: Any | None = None,
    ) -> MappingResolution:
        """
        Resolve canonical-to-source mapping from headers, overrides, and DB config.
        """

        source_headers = tuple(header for header in headers if header and header.strip())
        normalized_header_lookup: dict[str, str] = {
            normalize_header(header): header
            for header in source_headers
            if normalize_header(header)
        }
        if not source_headers:
            raise SchemaMappingError(
                message="CSV headers are empty; cannot resolve schema mapping.",
                errors=[
                    MappingErrorDetail(
                        code="empty_headers",
                        message="No CSV headers were provided.",
                    )
                ],
            )

        overrides = self._merge_overrides(
            mapping_config=mapping_config,
            manual_overrides=manual_overrides,
        )

        resolved: dict[str, str] = {}
        strategies: dict[str, str] = {}
        mapping_errors: list[MappingErrorDetail] = []

        for canonical_field, source_column in overrides.items():
            normalized_canonical = canonical_field.strip()
            if normalized_canonical not in CANONICAL_FIELDS:
                mapping_errors.append(
                    MappingErrorDetail(
                        code="invalid_override_field",
                        message="Manual override contains unknown canonical field.",
                        canonical_field=normalized_canonical,
                        source_column=source_column,
                    )
                )
                continue

            matched_source = normalized_header_lookup.get(normalize_header(source_column))
            if matched_source is None:
                mapping_errors.append(
                    MappingErrorDetail(
                        code="override_source_not_found",
                        message="Manual override points to a source column not present in CSV headers.",
                        canonical_field=normalized_canonical,
                        source_column=source_column,
                        context={"source_headers": list(source_headers)},
                    )
                )
                continue

            resolved[normalized_canonical] = matched_source
            strategies[normalized_canonical] = "override"

        used_headers = set(resolved.values())
        for canonical_field in CANONICAL_FIELDS:
            if canonical_field in resolved:
                continue

            exact = self._find_exact_or_alias_match(
                canonical_field=canonical_field,
                normalized_header_lookup=normalized_header_lookup,
            )
            if exact is not None and exact not in used_headers:
                resolved[canonical_field] = exact
                strategies[canonical_field] = "exact_or_alias"
                used_headers.add(exact)
                continue

            fuzzy_match = self._find_best_fuzzy_match(
                canonical_field=canonical_field,
                normalized_header_lookup=normalized_header_lookup,
                used_headers=used_headers,
                mapping_config=mapping_config,
            )
            if fuzzy_match is not None:
                resolved[canonical_field] = fuzzy_match
                strategies[canonical_field] = "fuzzy"
                used_headers.add(fuzzy_match)

        self._validator.validate(
            mapping=resolved,
            source_headers=source_headers,
            pre_errors=mapping_errors,
        )

        config_id = None
        if mapping_config is not None:
            raw_id = getattr(mapping_config, "id", None)
            config_id = str(raw_id) if raw_id is not None else None
        return MappingResolution(
            canonical_to_source=resolved,
            source_headers=source_headers,
            match_strategies=strategies,
            mapping_config_id=config_id,
        )

    def map_row(
        self,
        *,
        raw_row: Mapping[str, str | None],
        mapping: MappingResolution,
    ) -> dict[str, str | None]:
        """
        Map one source CSV row into canonical raw field values.
        """

        return {
            canonical_field: raw_row.get(source_column)
            for canonical_field, source_column in mapping.canonical_to_source.items()
        }

    @staticmethod
    def to_canonical_record(value: CanonicalInsightInput) -> CanonicalInsightRecord:
        """
        Convert validated canonical input into a canonical record model object.
        """

        return CanonicalInsightRecord(
            source_type=value.source_type,
            entity_name=value.entity_name,
            category=value.category,
            metric_name=value.metric_name,
            metric_value=value.metric_value,
            timestamp=value.timestamp,
            region=value.region,
            metadata_json=value.metadata_json,
        )

    def _find_exact_or_alias_match(
        self,
        *,
        canonical_field: str,
        normalized_header_lookup: Mapping[str, str],
    ) -> str | None:
        candidates = (
            canonical_field,
            *self._aliases.get(canonical_field, ()),
        )
        for candidate in candidates:
            match = normalized_header_lookup.get(normalize_header(candidate))
            if match:
                return match
        return None

    def _find_best_fuzzy_match(
        self,
        *,
        canonical_field: str,
        normalized_header_lookup: Mapping[str, str],
        used_headers: set[str],
        mapping_config: Any | None,
    ) -> str | None:
        alias_candidates = [canonical_field, *self._aliases.get(canonical_field, ())]
        alias_candidates.extend(self._config_aliases_for_field(mapping_config, canonical_field))
        normalized_candidates = [normalize_header(item) for item in alias_candidates if normalize_header(item)]
        if not normalized_candidates:
            return None

        best_header: str | None = None
        best_score = 0.0
        for header_norm, header_raw in normalized_header_lookup.items():
            if header_raw in used_headers:
                continue
            for candidate in normalized_candidates:
                score = SequenceMatcher(None, header_norm, candidate).ratio()
                if header_norm in candidate or candidate in header_norm:
                    score = max(score, 0.9)
                if score > best_score:
                    best_score = score
                    best_header = header_raw

        if best_header is not None and best_score >= self._fuzzy_threshold:
            return best_header
        return None

    @staticmethod
    def _merge_overrides(
        *,
        mapping_config: Any | None,
        manual_overrides: Mapping[str, str] | None,
    ) -> dict[str, str]:
        merged: dict[str, str] = {}

        if mapping_config is not None:
            config_mapping = getattr(mapping_config, "field_mapping_json", None)
            if isinstance(config_mapping, dict):
                for key, value in config_mapping.items():
                    if isinstance(key, str) and isinstance(value, str):
                        if key.strip() and value.strip():
                            merged[key.strip()] = value.strip()

        if manual_overrides:
            for key, value in manual_overrides.items():
                if key.strip() and value.strip():
                    merged[key.strip()] = value.strip()

        return merged

    @staticmethod
    def _config_aliases_for_field(mapping_config: Any | None, canonical_field: str) -> list[str]:
        if mapping_config is None:
            return []
        aliases_json = getattr(mapping_config, "alias_overrides_json", None)
        if not isinstance(aliases_json, dict):
            return []
        raw_aliases = aliases_json.get(canonical_field)
        if not isinstance(raw_aliases, list):
            return []
        return [alias for alias in raw_aliases if isinstance(alias, str) and alias.strip()]
