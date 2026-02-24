"""
app/mappers/canonical_mapper.py

Column mapping from arbitrary CSV headers to canonical fields.

Three-tier resolution (deterministic, no LLM):
  1. Exact case-insensitive match
  2. Alias dictionary lookup (static + category-aware + optional extras)
  3. Fuzzy match (rapidfuzz) only when score >= 92 and not ambiguous

Category-aware aliases allow metric names that only make sense in a
specific business domain (e.g. "mrr" -> metric_value when category=saas)
to be resolved automatically when a category hint is provided.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

_MIN_FUZZY_SIMILARITY = 92.0
_AMBIGUITY_RANGE = 5.0

# ---------------------------------------------------------------------------
# Canonical schema
# ---------------------------------------------------------------------------

CANONICAL_FIELDS: tuple[str, ...] = (
    "source_type",
    "entity_name",
    "category",
    "role",
    "metric_name",
    "metric_value",
    "timestamp",
    "region",
    "metadata_json",
)

REQUIRED_CANONICAL_FIELDS: tuple[str, ...] = (
    "entity_name",
    "category",
    "metric_name",
    "metric_value",
)

# ---------------------------------------------------------------------------
# Static aliases (category-independent)
# ---------------------------------------------------------------------------

DEFAULT_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "source_type": (
        "source", "source_kind", "source kind", "input_source",
        "data_source", "origin",
    ),
    "entity_name": (
        "entity", "company", "client", "competitor", "account_name",
        "business_name", "brand", "organization", "organisation",
        "firm", "vendor", "tenant", "customer_name",
    ),
    "category": (
        "insight_category", "metric_category", "domain", "topic",
        "business_area", "department", "segment", "vertical",
    ),
    "role": (
        "org_role", "organization_role", "job_role", "function",
        "persona", "team_role", "position", "owner_role",
    ),
    "metric_name": (
        "metric", "kpi", "measure_name", "metric_key",
        "indicator", "measure", "signal", "kpi_name",
    ),
    "metric_value": (
        "value", "metric_amount", "measure_value", "kpi_value",
        "amount", "total", "count", "quantity", "score",
        "number", "figure", "result",
    ),
    "timestamp": (
        "date", "datetime", "event_time", "measured_at", "recorded_at",
        "month", "period", "report_date", "created_date", "time",
        "observation_date", "data_date", "as_of_date", "effective_date",
    ),
    "region": (
        "geo", "geography", "country", "market_region", "location",
        "territory", "market", "area", "state", "city", "zone",
    ),
    "metadata_json": (
        "metadata", "meta", "context_json", "attributes",
        "extra", "tags", "properties", "details",
    ),
}

# ---------------------------------------------------------------------------
# Category-aware aliases
#
# These map a (category, source_header) pair to a canonical field.
# When a category hint is provided, these are checked BEFORE static
# aliases, giving domain-specific headers priority.
#
# Format: { category: { canonical_field: (alias, ...) } }
# ---------------------------------------------------------------------------

CATEGORY_AWARE_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "saas": {
        "metric_value": (
            "mrr", "arr", "monthly_recurring_revenue",
            "annual_recurring_revenue", "acv", "tcv",
        ),
        "metric_name": (
            "saas_metric", "subscription_metric",
        ),
        "entity_name": (
            "account", "workspace", "subscription",
        ),
        "timestamp": (
            "billing_date", "renewal_date", "subscription_start",
        ),
    },
    "ecommerce": {
        "metric_value": (
            "revenue", "gmv", "gross_merchandise_value",
            "order_value", "transaction_amount", "sales",
            "net_revenue", "gross_revenue",
        ),
        "metric_name": (
            "product_metric", "sku_metric", "order_metric",
        ),
        "entity_name": (
            "store", "shop", "merchant", "seller",
        ),
        "timestamp": (
            "order_date", "transaction_date", "purchase_date",
        ),
    },
    "agency": {
        "metric_value": (
            "total_revenue", "billable_amount", "retainer",
            "project_value", "contract_value",
        ),
        "metric_name": (
            "engagement_metric", "campaign_metric",
        ),
        "entity_name": (
            "agency", "account_name", "client_name",
        ),
        "timestamp": (
            "campaign_date", "report_period", "invoice_date",
        ),
    },
    "finance": {
        "metric_value": (
            "aum", "nav", "portfolio_value", "loan_amount",
            "premium", "interest_income",
        ),
        "entity_name": (
            "fund", "portfolio", "borrower", "policyholder",
        ),
    },
    "healthcare": {
        "metric_value": (
            "patient_count", "claim_amount", "reimbursement",
            "bed_occupancy", "readmission_rate",
        ),
        "entity_name": (
            "hospital", "clinic", "provider", "facility",
        ),
    },
    "retail": {
        "metric_value": (
            "sales", "revenue", "footfall", "basket_size",
            "same_store_sales", "units_sold",
        ),
        "entity_name": (
            "store", "outlet", "branch", "location_name",
        ),
        "timestamp": (
            "sale_date", "transaction_date",
        ),
    },
    "logistics": {
        "metric_value": (
            "shipment_volume", "delivery_count", "freight_cost",
            "on_time_rate", "utilization",
        ),
        "entity_name": (
            "carrier", "warehouse", "hub", "depot",
        ),
    },
}


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_header(header: str) -> str:
    """Normalize a column name: lowercase, strip non-alphanumeric."""
    return "".join(ch for ch in header.strip().lower() if ch.isalnum())


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MissingRequiredColumnsError(ValueError):
    """Raised when required canonical fields cannot be mapped from CSV headers."""

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


class AmbiguousColumnMatchError(ValueError):
    """Raised when fuzzy matching finds competing candidates too close in score."""

    def __init__(self, *, errors: Sequence[dict[str, Any]]) -> None:
        fields = ", ".join(sorted({str(e.get("canonical_field", "")) for e in errors}))
        message = (
            "Ambiguous column mapping detected. "
            f"Manual mapping required for fields: {fields}."
        )
        super().__init__(message)
        self.errors = tuple(errors)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnMapping:
    """Resolved mapping between canonical field names and source CSV headers."""

    canonical_to_source: dict[str, str]
    source_headers: tuple[str, ...]
    match_strategies: dict[str, str] = field(default_factory=dict)
    confidence_scores: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Mapper
# ---------------------------------------------------------------------------


class ColumnMapper:
    """
    Maps incoming CSV columns to canonical schema fields.

    Resolution order (per canonical field):
      1. Exact case-insensitive match against the canonical field name
      2. Alias dictionary lookup (category-aware + static + extra aliases)
      3. Fuzzy match via rapidfuzz only when:
         - score >= 92
         - no competing match within 5 score points

    Each header is consumed at most once (first-match-wins, no double-mapping).
    """

    def __init__(
        self,
        aliases: Mapping[str, Sequence[str]] | None = None,
        *,
        fuzzy_threshold: float = 80.0,
        category_aliases: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    ) -> None:
        self._aliases: dict[str, tuple[str, ...]] = {
            canonical: tuple(values)
            for canonical, values in (aliases or DEFAULT_COLUMN_ALIASES).items()
        }
        self._category_aliases: dict[str, dict[str, tuple[str, ...]]] = {
            cat: {
                field: tuple(als)
                for field, als in fields.items()
            }
            for cat, fields in (category_aliases or CATEGORY_AWARE_ALIASES).items()
        }
        self._fuzzy_threshold = max(
            _MIN_FUZZY_SIMILARITY,
            min(100.0, fuzzy_threshold),
        )

    def build_mapping(
        self,
        headers: Sequence[str],
        *,
        required_fields: Sequence[str] = REQUIRED_CANONICAL_FIELDS,
        category_hint: str | None = None,
        extra_aliases: Mapping[str, Sequence[str]] | None = None,
    ) -> ColumnMapping:
        """Resolve canonical field -> source-header mapping for one CSV."""
        ci_lookup: dict[str, str] = {}
        normalized_lookup: dict[str, str] = {}
        for header in headers:
            clean = header.strip()
            if not clean:
                continue
            lower = clean.lower()
            if lower not in ci_lookup:
                ci_lookup[lower] = clean
            norm = normalize_header(clean)
            if norm and norm not in normalized_lookup:
                normalized_lookup[norm] = clean

        merged_aliases = self._merge_aliases(category_hint, extra_aliases)

        mapping: dict[str, str] = {}
        strategies: dict[str, str] = {}
        confidence_scores: dict[str, float] = {}
        used_headers: set[str] = set()
        ambiguous_errors: list[dict[str, Any]] = []

        for canonical_field in CANONICAL_FIELDS:
            exact = self._exact_case_insensitive_match(
                canonical_field=canonical_field,
                ci_lookup=ci_lookup,
                used_headers=used_headers,
            )
            if exact is not None:
                mapping[canonical_field] = exact
                strategies[canonical_field] = "exact"
                confidence_scores[canonical_field] = 1.0
                used_headers.add(exact)
                logger.info(
                    "SchemaMapper decision field=%s source=%s stage=exact confidence=%.2f",
                    canonical_field,
                    exact,
                    1.0,
                )
                continue

            aliases = self._build_alias_candidates(
                canonical_field=canonical_field,
                merged_aliases=merged_aliases,
                category_hint=category_hint,
            )
            alias_match = self._alias_lookup_match(
                aliases=aliases,
                ci_lookup=ci_lookup,
                used_headers=used_headers,
            )
            if alias_match is not None:
                mapping[canonical_field] = alias_match
                strategies[canonical_field] = "alias"
                confidence_scores[canonical_field] = 0.99
                used_headers.add(alias_match)
                logger.info(
                    "SchemaMapper decision field=%s source=%s stage=alias confidence=%.2f",
                    canonical_field,
                    alias_match,
                    0.99,
                )
                continue

            fuzzy_result = self._strict_fuzzy_match(
                canonical_field=canonical_field,
                candidates=[canonical_field, *aliases],
                normalized_lookup=normalized_lookup,
                used_headers=used_headers,
            )
            if fuzzy_result is None:
                logger.info(
                    "SchemaMapper decision field=%s source=<none> stage=unresolved",
                    canonical_field,
                )
                continue

            if fuzzy_result["ambiguous"]:
                ambiguous_errors.append(
                    {
                        "code": "ambiguous_fuzzy_match",
                        "canonical_field": canonical_field,
                        "message": (
                            "Multiple fuzzy candidates are within the ambiguity range."
                        ),
                        "context": {
                            "best_match": fuzzy_result["best_match"],
                            "best_score": fuzzy_result["best_score"],
                            "near_matches": fuzzy_result["near_matches"],
                            "threshold": self._fuzzy_threshold,
                            "ambiguity_range": _AMBIGUITY_RANGE,
                        },
                    }
                )
                logger.warning(
                    "SchemaMapper decision field=%s stage=fuzzy_ambiguous best=%s score=%.2f",
                    canonical_field,
                    fuzzy_result["best_match"],
                    fuzzy_result["best_score"],
                )
                continue

            mapped_header = str(fuzzy_result["best_match"])
            mapped_score = float(fuzzy_result["best_score"])
            mapping[canonical_field] = mapped_header
            strategies[canonical_field] = "fuzzy"
            confidence_scores[canonical_field] = round(mapped_score / 100.0, 4)
            used_headers.add(mapped_header)
            logger.info(
                "SchemaMapper decision field=%s source=%s stage=fuzzy confidence=%.4f score=%.2f",
                canonical_field,
                mapped_header,
                confidence_scores[canonical_field],
                mapped_score,
            )

        if ambiguous_errors:
            raise AmbiguousColumnMatchError(errors=ambiguous_errors)

        missing = [f for f in required_fields if f not in mapping]
        if missing:
            raise MissingRequiredColumnsError(missing=missing, headers=headers)

        return ColumnMapping(
            canonical_to_source=mapping,
            source_headers=tuple(headers),
            match_strategies=strategies,
            confidence_scores=confidence_scores,
        )

    def map_row(
        self,
        *,
        raw_row: Mapping[str, str | None],
        mapping: ColumnMapping,
    ) -> dict[str, str | None]:
        """Convert a source CSV row into canonical raw fields."""
        return {
            canonical_field: raw_row.get(source_column)
            for canonical_field, source_column in mapping.canonical_to_source.items()
        }

    # ------------------------------------------------------------------
    # Candidate list construction
    # ------------------------------------------------------------------

    def _build_alias_candidates(
        self,
        canonical_field: str,
        merged_aliases: dict[str, list[str]],
        category_hint: str | None,
    ) -> list[str]:
        """Build ordered alias candidates for Stage 2 lookup."""
        candidates: list[str] = []

        # Category-aware aliases first (higher priority)
        if category_hint:
            cat_lower = category_hint.strip().lower()
            cat_fields = self._category_aliases.get(cat_lower, {})
            candidates.extend(cat_fields.get(canonical_field, ()))

        # Static + extra aliases
        candidates.extend(merged_aliases.get(canonical_field, []))

        return candidates

    def _merge_aliases(
        self,
        category_hint: str | None,
        extra_aliases: Mapping[str, Sequence[str]] | None,
    ) -> dict[str, list[str]]:
        """Merge static aliases with optional extra aliases."""
        merged: dict[str, list[str]] = {
            field: list(aliases) for field, aliases in self._aliases.items()
        }
        if extra_aliases:
            for field, aliases in extra_aliases.items():
                existing = merged.get(field, [])
                existing.extend(a for a in aliases if a not in existing)
                merged[field] = existing
        return merged

    # ------------------------------------------------------------------
    # Match strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _exact_case_insensitive_match(
        *,
        canonical_field: str,
        ci_lookup: Mapping[str, str],
        used_headers: set[str],
    ) -> str | None:
        """Stage 1: exact case-insensitive match against canonical field name."""
        raw_header = ci_lookup.get(canonical_field.strip().lower())
        if raw_header is not None and raw_header not in used_headers:
            return raw_header
        return None

    @staticmethod
    def _alias_lookup_match(
        *,
        aliases: Sequence[str],
        ci_lookup: Mapping[str, str],
        used_headers: set[str],
    ) -> str | None:
        """Stage 2: alias dictionary lookup (case-insensitive)."""
        for alias in aliases:
            raw_header = ci_lookup.get(alias.strip().lower())
            if raw_header is not None and raw_header not in used_headers:
                return raw_header
        return None

    def _strict_fuzzy_match(
        self,
        *,
        canonical_field: str,
        candidates: Sequence[str],
        normalized_lookup: Mapping[str, str],
        used_headers: set[str],
    ) -> dict[str, Any] | None:
        """Stage 3: fuzzy match with strict threshold and ambiguity guard."""
        available: dict[str, str] = {
            norm: raw
            for norm, raw in normalized_lookup.items()
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
        header_scores: dict[str, float] = {}
        for candidate in normalized_candidates:
            results = process.extract(
                candidate,
                available_norms,
                scorer=fuzz.token_sort_ratio,
                limit=len(available_norms),
            )
            for match_norm, score, _ in results:
                header = available[match_norm]
                prev = header_scores.get(header, 0.0)
                if score > prev:
                    header_scores[header] = float(score)

        if not header_scores:
            return None
        ranked = sorted(
            header_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        best_header, best_score = ranked[0]
        if best_score < self._fuzzy_threshold:
            return None

        near_matches = [
            {"header": header, "score": round(score, 2)}
            for header, score in ranked[1:]
            if score >= (best_score - _AMBIGUITY_RANGE)
        ]
        if near_matches:
            return {
                "canonical_field": canonical_field,
                "ambiguous": True,
                "best_match": best_header,
                "best_score": round(best_score, 2),
                "near_matches": near_matches,
            }

        return {
            "canonical_field": canonical_field,
            "ambiguous": False,
            "best_match": best_header,
            "best_score": round(best_score, 2),
            "near_matches": [],
        }

    # ------------------------------------------------------------------
    # Public accessors for introspection
    # ------------------------------------------------------------------

    @property
    def fuzzy_threshold(self) -> float:
        return self._fuzzy_threshold

    @property
    def static_aliases(self) -> dict[str, tuple[str, ...]]:
        return dict(self._aliases)

    @property
    def category_aliases(self) -> dict[str, dict[str, tuple[str, ...]]]:
        return dict(self._category_aliases)

