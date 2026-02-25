"""
app/mappers/schema_interpreter.py

Deterministic candidate scoring engine for schema mapping.

Scores each source column against canonical fields using a weighted formula:
    score = 0.45*alias_exact + 0.25*fuzzy_similarity + 0.20*dtype_compat + 0.10*value_pattern

Thresholds:
    >= 0.90  → auto_map
    0.75–0.89 → map_with_warning
    < 0.75   → unmapped

Every decision is recorded as a typed MappingDecision with full rule trace.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from rapidfuzz import fuzz

from app.mappers.canonical_mapper import (
    CANONICAL_FIELDS,
    CATEGORY_AWARE_ALIASES,
    DEFAULT_COLUMN_ALIASES,
    REQUIRED_CANONICAL_FIELDS,
    normalize_header,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------

_W_ALIAS_EXACT: float = 0.45
_W_FUZZY: float = 0.25
_W_DTYPE: float = 0.20
_W_PATTERN: float = 0.10

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_AUTO_MAP_THRESHOLD: float = 0.90
_WARNING_THRESHOLD: float = 0.75

# ---------------------------------------------------------------------------
# Dtype expectations per canonical field
# ---------------------------------------------------------------------------

_EXPECTED_DTYPES: dict[str, str] = {
    "source_type": "string",
    "entity_name": "string",
    "category": "string",
    "role": "string",
    "metric_name": "string",
    "metric_value": "numeric",
    "timestamp": "datetime",
    "region": "string",
    "metadata_json": "json",
}

# ---------------------------------------------------------------------------
# Value pattern regexes
# ---------------------------------------------------------------------------

_NUMERIC_RE = re.compile(r"^-?\d[\d,]*\.?\d*$")
_ISO_DATE_RE = re.compile(
    r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}"
)
_JSON_RE = re.compile(r"^\s*[\[{]")
_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# ---------------------------------------------------------------------------
# Typed output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MappingDecision:
    """One scored mapping decision between a source column and a canonical field."""

    source_column: str
    canonical_field: str
    confidence: float
    ambiguity_gap: float
    matched_rule: str
    scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class InterpretationResult:
    """Full interpretation result for a set of source columns."""

    decisions: tuple[MappingDecision, ...]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    audit: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _alias_exact_score(
    source_norm: str,
    canonical_field: str,
    aliases: Mapping[str, Sequence[str]],
    category_aliases: dict[str, tuple[str, ...]],
) -> float:
    """Return 1.0 for exact alias/field match, 0.0 otherwise."""
    if source_norm == normalize_header(canonical_field):
        return 1.0

    for alias in category_aliases.get(canonical_field, ()):
        if source_norm == normalize_header(alias):
            return 1.0

    for alias in aliases.get(canonical_field, ()):
        if source_norm == normalize_header(alias):
            return 1.0

    return 0.0


def _fuzzy_score(source_norm: str, canonical_field: str, aliases: Sequence[str]) -> float:
    """Best fuzzy similarity (0–1) across field name and all aliases."""
    candidates = [normalize_header(canonical_field)]
    candidates.extend(normalize_header(a) for a in aliases if normalize_header(a))

    best = 0.0
    for candidate in candidates:
        token_sort = fuzz.token_sort_ratio(source_norm, candidate) / 100.0
        partial = fuzz.partial_ratio(source_norm, candidate) / 100.0
        best = max(best, token_sort, partial)
    return best


def _dtype_compatibility(
    canonical_field: str,
    sample_values: Sequence[str],
) -> float:
    """Score dtype compatibility between sample values and expected type."""
    expected = _EXPECTED_DTYPES.get(canonical_field, "string")
    if not sample_values:
        return 0.5  # no evidence — neutral

    non_empty = [v for v in sample_values if v.strip()]
    if not non_empty:
        return 0.5

    matches = 0
    for val in non_empty:
        val = val.strip()
        if expected == "numeric":
            if _NUMERIC_RE.match(val.replace(",", "")):
                matches += 1
        elif expected == "datetime":
            if _ISO_DATE_RE.match(val):
                matches += 1
        elif expected == "json":
            if _JSON_RE.match(val):
                matches += 1
        elif expected == "string":
            if (
                not _NUMERIC_RE.match(val.replace(",", ""))
                and not _JSON_RE.match(val)
                and not _ISO_DATE_RE.match(val)
            ):
                matches += 1

    return matches / len(non_empty)


def _value_pattern_fit(
    canonical_field: str,
    sample_values: Sequence[str],
) -> float:
    """Heuristic pattern fitness for the canonical field."""
    if not sample_values:
        return 0.5

    non_empty = [v.strip() for v in sample_values if v.strip()]
    if not non_empty:
        return 0.5

    if canonical_field in ("source_type", "category", "role"):
        unique_ratio = len(set(v.lower() for v in non_empty)) / max(len(non_empty), 1)
        if unique_ratio <= 0.3:
            return 1.0
        if unique_ratio <= 0.6:
            return 0.6
        return 0.3

    if canonical_field == "metric_name":
        slug_count = sum(1 for v in non_empty if _SLUG_RE.match(v.lower().replace(" ", "_")))
        return slug_count / len(non_empty)

    if canonical_field == "entity_name":
        return 0.7 if len(set(non_empty)) >= 2 else 0.5

    if canonical_field == "metric_value":
        numeric_count = sum(1 for v in non_empty if _NUMERIC_RE.match(v.replace(",", "")))
        return numeric_count / len(non_empty)

    if canonical_field == "timestamp":
        date_count = sum(1 for v in non_empty if _ISO_DATE_RE.match(v))
        return date_count / len(non_empty)

    return 0.5


# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------


class SchemaInterpreter:
    """
    Deterministic schema interpreter with scored candidate mapping.

    Produces a MappingDecision per source column with full rule traces.
    Rejects silent low-confidence mappings; surfaces explicit warnings/errors.
    """

    def __init__(
        self,
        *,
        aliases: Mapping[str, Sequence[str]] | None = None,
        category_hint: str | None = None,
    ) -> None:
        self._aliases: dict[str, tuple[str, ...]] = {
            canonical: tuple(values)
            for canonical, values in (aliases or DEFAULT_COLUMN_ALIASES).items()
        }
        self._category_hint = (category_hint or "").strip().lower() or None

    def interpret(
        self,
        headers: Sequence[str],
        *,
        sample_rows: Sequence[Mapping[str, str]] | None = None,
        category_hint: str | None = None,
    ) -> InterpretationResult:
        """
        Score every (source_column, canonical_field) pair and produce decisions.

        Args:
            headers:       Source column headers.
            sample_rows:   Optional sample data rows for dtype/pattern scoring.
            category_hint: Business category for domain-specific aliases.

        Returns:
            InterpretationResult with decisions, warnings, errors, and audit metadata.
        """
        active_category = (
            (category_hint or self._category_hint or "").strip().lower() or None
        )
        cat_aliases: dict[str, tuple[str, ...]] = {}
        if active_category:
            raw = CATEGORY_AWARE_ALIASES.get(active_category, {})
            cat_aliases = {k: tuple(v) for k, v in raw.items()}

        samples = list(sample_rows or [])
        score_matrix: dict[str, dict[str, float]] = {}
        detail_matrix: dict[str, dict[str, dict[str, float]]] = {}

        for header in headers:
            header_clean = header.strip()
            if not header_clean:
                continue

            source_norm = normalize_header(header_clean)
            if not source_norm:
                continue

            sample_values = [
                str(row.get(header, "")) for row in samples
            ]

            field_scores: dict[str, float] = {}
            field_details: dict[str, dict[str, float]] = {}

            for canonical_field in CANONICAL_FIELDS:
                all_aliases: list[str] = list(cat_aliases.get(canonical_field, ()))
                all_aliases.extend(self._aliases.get(canonical_field, ()))

                alias_ex = _alias_exact_score(
                    source_norm, canonical_field, self._aliases, cat_aliases,
                )
                fuzzy_sim = _fuzzy_score(source_norm, canonical_field, all_aliases)
                dtype_compat = _dtype_compatibility(canonical_field, sample_values)
                pattern_fit = _value_pattern_fit(canonical_field, sample_values)

                score = (
                    _W_ALIAS_EXACT * alias_ex
                    + _W_FUZZY * fuzzy_sim
                    + _W_DTYPE * dtype_compat
                    + _W_PATTERN * pattern_fit
                )

                field_scores[canonical_field] = round(score, 4)
                field_details[canonical_field] = {
                    "alias_exact": round(alias_ex, 4),
                    "fuzzy_similarity": round(fuzzy_sim, 4),
                    "dtype_compatibility": round(dtype_compat, 4),
                    "value_pattern_fit": round(pattern_fit, 4),
                }

            score_matrix[header_clean] = field_scores
            detail_matrix[header_clean] = field_details

        # Greedy assignment: pick best unambiguous pairs
        decisions: list[MappingDecision] = []
        warnings: list[str] = []
        errors: list[str] = []
        used_sources: set[str] = set()
        used_targets: set[str] = set()

        # Build flat list of (score, header, field) sorted descending
        candidates: list[tuple[float, str, str]] = []
        for header, field_scores in score_matrix.items():
            for canonical_field, score in field_scores.items():
                candidates.append((score, header, canonical_field))
        candidates.sort(key=lambda x: x[0], reverse=True)

        for score, header, canonical_field in candidates:
            if header in used_sources or canonical_field in used_targets:
                continue

            details = detail_matrix[header][canonical_field]

            # Compute ambiguity gap: distance to next-best competing score
            # for this header targeting ANY remaining canonical field
            remaining_scores = [
                score_matrix[header][cf]
                for cf in CANONICAL_FIELDS
                if cf != canonical_field and cf not in used_targets
            ]
            second_best = max(remaining_scores) if remaining_scores else 0.0
            ambiguity_gap = round(score - second_best, 4)

            # Also check if another unused header has a close score for same field
            competing_headers = [
                (score_matrix[h][canonical_field], h)
                for h in score_matrix
                if h != header and h not in used_sources
            ]
            if competing_headers:
                best_competitor_score = max(s for s, _ in competing_headers)
                header_gap = score - best_competitor_score
                ambiguity_gap = round(min(ambiguity_gap, header_gap), 4)

            if score >= _AUTO_MAP_THRESHOLD:
                rule = "auto_map"
            elif score >= _WARNING_THRESHOLD:
                rule = "map_with_warning"
                warnings.append(
                    f"Column '{header}' mapped to '{canonical_field}' with moderate "
                    f"confidence ({score:.2f}). Verify mapping is correct."
                )
            else:
                rule = "unmapped"
                continue  # do NOT assign low-confidence mappings

            decision = MappingDecision(
                source_column=header,
                canonical_field=canonical_field,
                confidence=score,
                ambiguity_gap=ambiguity_gap,
                matched_rule=rule,
                scores=details,
            )
            decisions.append(decision)
            used_sources.add(header)
            used_targets.add(canonical_field)

        # Check required fields that ended up unmapped
        mapped_fields = {d.canonical_field for d in decisions}
        for req in REQUIRED_CANONICAL_FIELDS:
            if req not in mapped_fields:
                errors.append(
                    f"Required canonical field '{req}' could not be mapped. "
                    "No source column scored above the 0.75 threshold."
                )

        # Build audit metadata
        audit: dict[str, Any] = {
            "interpreted_at": datetime.now(tz=timezone.utc).isoformat(),
            "category_hint": active_category,
            "source_column_count": len(headers),
            "canonical_field_count": len(CANONICAL_FIELDS),
            "auto_mapped": sum(1 for d in decisions if d.matched_rule == "auto_map"),
            "mapped_with_warning": sum(
                1 for d in decisions if d.matched_rule == "map_with_warning"
            ),
            "unmapped_fields": sorted(
                f for f in CANONICAL_FIELDS if f not in mapped_fields
            ),
            "score_matrix": {
                header: {
                    cf: {
                        "composite": score_matrix[header][cf],
                        **detail_matrix[header][cf],
                    }
                    for cf in CANONICAL_FIELDS
                }
                for header in score_matrix
            },
        }

        return InterpretationResult(
            decisions=tuple(decisions),
            warnings=tuple(warnings),
            errors=tuple(errors),
            audit=audit,
        )

    def to_mapping_dict(
        self,
        result: InterpretationResult,
    ) -> dict[str, str]:
        """Convert decisions to canonical_field → source_column dict."""
        return {
            d.canonical_field: d.source_column
            for d in result.decisions
        }
