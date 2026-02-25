"""
app/mappers/schema_mapper.py

Dynamic schema mapping engine for CSV-to-canonical mapping.

Delegates fuzzy matching to rapidfuzz (via canonical_mapper utilities)
and adds mapping_config DB integration, override merging, and
MappingValidator enforcement on top.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from rapidfuzz import fuzz, process

from app.domain.canonical_insight import CanonicalInsightInput
from app.mappers.canonical_mapper import (
    CANONICAL_FIELDS,
    CATEGORY_AWARE_ALIASES,
    DEFAULT_COLUMN_ALIASES,
    REQUIRED_CANONICAL_FIELDS,
    normalize_header,
)
from app.mappers.schema_interpreter import (
    InterpretationResult,
    MappingDecision,
    SchemaInterpreter,
)
from app.validators.mapping_validator import MappingErrorDetail, MappingValidator, SchemaMappingError
from db.models.canonical_insight_record import CanonicalInsightRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MappingResolution:
    """Final resolved mapping metadata."""

    canonical_to_source: dict[str, str]
    source_headers: tuple[str, ...]
    match_strategies: dict[str, str]
    mapping_config_id: str | None = None
    interpretation: InterpretationResult | None = None


class SchemaMapper:
    """
    Resolves source CSV schemas into canonical field mappings.

    Resolution order per canonical field:
      1. Manual / DB-config overrides
      2. Exact normalized match (field name + static aliases + category aliases)
      3. Fuzzy match (rapidfuzz token_sort_ratio + partial_ratio)
    """

    def __init__(
        self,
        *,
        aliases: Mapping[str, Sequence[str]] | None = None,
        validator: MappingValidator | None = None,
        fuzzy_threshold: float = 80.0,
        category_hint: str | None = None,
    ) -> None:
        self._aliases: dict[str, tuple[str, ...]] = {
            canonical: tuple(values)
            for canonical, values in (aliases or DEFAULT_COLUMN_ALIASES).items()
        }
        self._validator = validator or MappingValidator(
            required_fields=REQUIRED_CANONICAL_FIELDS,
            canonical_fields=CANONICAL_FIELDS,
        )
        self._fuzzy_threshold = max(0.0, min(100.0, fuzzy_threshold))
        self._category_hint = category_hint
        self._interpreter = SchemaInterpreter(
            aliases=aliases,
            category_hint=category_hint,
        )

    def resolve_mapping(
        self,
        headers: Sequence[str],
        *,
        manual_overrides: Mapping[str, str] | None = None,
        mapping_config: Any | None = None,
        category_hint: str | None = None,
        sample_rows: Sequence[Mapping[str, str]] | None = None,
    ) -> MappingResolution:
        """
        Resolve canonical-to-source mapping from headers, overrides, and DB config.

        Args:
            headers:          Raw CSV column headers.
            manual_overrides: Explicit canonical→source column overrides.
            mapping_config:   DB MappingConfig object (has field_mapping_json,
                              alias_overrides_json).
            category_hint:    Business category to activate domain-specific aliases.
            sample_rows:      Optional sample data rows for interpreter scoring.
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

        # Phase 0: Apply explicit overrides
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

        # Resolve category hint (parameter > constructor > None)
        active_category = (category_hint or self._category_hint or "").strip().lower() or None

        used_headers = set(resolved.values())
        for canonical_field in CANONICAL_FIELDS:
            if canonical_field in resolved:
                continue

            candidates = self._build_candidates(
                canonical_field, mapping_config, active_category
            )

            # Phase 1: Exact/alias match
            exact = self._find_exact_or_alias_match(
                candidates=candidates,
                normalized_header_lookup=normalized_header_lookup,
                used_headers=used_headers,
            )
            if exact is not None:
                resolved[canonical_field] = exact
                strategies[canonical_field] = "exact_or_alias"
                used_headers.add(exact)
                continue

            # Phase 2: Fuzzy match (rapidfuzz)
            fuzzy_match = self._find_best_fuzzy_match(
                candidates=candidates,
                normalized_header_lookup=normalized_header_lookup,
                used_headers=used_headers,
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

        # Run interpreter for audit/scoring metadata
        active_category = (category_hint or self._category_hint or "").strip().lower() or None
        interpretation = self._interpreter.interpret(
            headers,
            sample_rows=sample_rows,
            category_hint=active_category,
        )

        # Log interpreter warnings
        for warning in interpretation.warnings:
            logger.warning("SchemaInterpreter: %s", warning)
        for error in interpretation.errors:
            logger.error("SchemaInterpreter: %s", error)

        return MappingResolution(
            canonical_to_source=resolved,
            source_headers=source_headers,
            match_strategies=strategies,
            mapping_config_id=config_id,
            interpretation=interpretation,
        )

    def map_row(
        self,
        *,
        raw_row: Mapping[str, str | None],
        mapping: MappingResolution,
    ) -> dict[str, str | None]:
        """Map one source CSV row into canonical raw field values."""
        return {
            canonical_field: raw_row.get(source_column)
            for canonical_field, source_column in mapping.canonical_to_source.items()
        }

    @staticmethod
    def to_canonical_record(value: CanonicalInsightInput) -> CanonicalInsightRecord:
        """Convert validated canonical input into a canonical record model object."""
        return CanonicalInsightRecord(
            source_type=value.source_type,
            entity_name=value.entity_name,
            category=value.category,
            role=value.role,
            metric_name=value.metric_name,
            metric_value=value.metric_value,
            timestamp=value.timestamp,
            region=value.region,
            metadata_json=value.metadata_json,
        )

    # ------------------------------------------------------------------
    # Candidate construction
    # ------------------------------------------------------------------

    def _build_candidates(
        self,
        canonical_field: str,
        mapping_config: Any | None,
        category: str | None,
    ) -> list[str]:
        """Build ordered candidate list for one canonical field.

        Order: field name → category aliases → static aliases → DB config aliases.
        """
        candidates: list[str] = [canonical_field]

        # Category-aware aliases (highest priority after field name)
        if category:
            cat_fields = CATEGORY_AWARE_ALIASES.get(category, {})
            candidates.extend(cat_fields.get(canonical_field, ()))

        # Static aliases
        candidates.extend(self._aliases.get(canonical_field, ()))

        # DB mapping config alias overrides
        candidates.extend(self._config_aliases_for_field(mapping_config, canonical_field))

        return candidates

    # ------------------------------------------------------------------
    # Match strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _find_exact_or_alias_match(
        *,
        candidates: Sequence[str],
        normalized_header_lookup: Mapping[str, str],
        used_headers: set[str],
    ) -> str | None:
        """Try exact normalized match against ordered candidate list."""
        for candidate in candidates:
            match = normalized_header_lookup.get(normalize_header(candidate))
            if match is not None and match not in used_headers:
                return match
        return None

    def _find_best_fuzzy_match(
        self,
        *,
        candidates: Sequence[str],
        normalized_header_lookup: Mapping[str, str],
        used_headers: set[str],
    ) -> str | None:
        """Find best fuzzy match using rapidfuzz.

        Uses token_sort_ratio for word-order resilience and partial_ratio
        for substring containment. Best score across both scorers wins.
        """
        available: dict[str, str] = {
            norm: raw
            for norm, raw in normalized_header_lookup.items()
            if raw not in used_headers
        }
        if not available:
            return None

        normalized_candidates = [
            normalize_header(c) for c in candidates if normalize_header(c)
        ]
        if not normalized_candidates:
            return None

        available_norms = list(available.keys())
        best_header: str | None = None
        best_score: float = 0.0

        for candidate in normalized_candidates:
            for scorer in (fuzz.token_sort_ratio, fuzz.partial_ratio):
                result = process.extractOne(
                    candidate,
                    available_norms,
                    scorer=scorer,
                    score_cutoff=self._fuzzy_threshold,
                )
                if result is not None:
                    match_norm, score, _ = result
                    if score > best_score:
                        best_score = score
                        best_header = available[match_norm]

        return best_header

    # ------------------------------------------------------------------
    # Override / config helpers
    # ------------------------------------------------------------------

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
