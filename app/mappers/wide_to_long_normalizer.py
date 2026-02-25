"""
app/mappers/wide_to_long_normalizer.py

Reusable wide-to-long (unpivot) normalizer for CSV and API ingestion.

Detects wide metric columns whose names encode a metric + temporal suffix
(e.g. ``revenue_2024_Q1``, ``churn_Jan_2025``) and melts them into canonical
long rows with explicit ``metric_name``, ``metric_value``, and ``timestamp``
columns.

Detection is fully deterministic — no LLM calls.  Parsing relies on regex
patterns for quarterly, monthly, and YYYY date fragments.

The module is integrated **before** canonical validation so that both wide
and long datasets flow through the same downstream pipeline.
"""

from __future__ import annotations

import calendar
import logging
import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Temporal pattern library
# ---------------------------------------------------------------------------

# Month abbreviation → month number
_MONTH_ABBR: dict[str, int] = {
    name.lower(): idx for idx, name in enumerate(calendar.month_abbr) if name
}
# Full month name → month number
_MONTH_FULL: dict[str, int] = {
    name.lower(): idx for idx, name in enumerate(calendar.month_name) if name
}

# Compile joined alternatives once
_MONTH_ABBR_PAT = "|".join(calendar.month_abbr[1:])  # Jan|Feb|…
_MONTH_FULL_PAT = "|".join(calendar.month_name[1:])   # January|February|…

# Patterns tried against each column name (case-insensitive).
# Named groups: ``metric`` (prefix), ``year``, ``quarter``, ``month``.
_TEMPORAL_PATTERNS: list[re.Pattern[str]] = [
    # revenue_2024_Q1 / revenue_2024_q3
    re.compile(
        r"^(?P<metric>.+?)[_\s-](?P<year>\d{4})[_\s-]Q(?P<quarter>[1-4])$",
        re.IGNORECASE,
    ),
    # revenue_Q1_2024
    re.compile(
        r"^(?P<metric>.+?)[_\s-]Q(?P<quarter>[1-4])[_\s-](?P<year>\d{4})$",
        re.IGNORECASE,
    ),
    # churn_Jan_2025 / churn_January_2025
    re.compile(
        rf"^(?P<metric>.+?)[_\s-](?P<month>{_MONTH_ABBR_PAT})[_\s-](?P<year>\d{{4}})$",
        re.IGNORECASE,
    ),
    re.compile(
        rf"^(?P<metric>.+?)[_\s-](?P<month>{_MONTH_FULL_PAT})[_\s-](?P<year>\d{{4}})$",
        re.IGNORECASE,
    ),
    # churn_2025_Jan / churn_2025_February
    re.compile(
        rf"^(?P<metric>.+?)[_\s-](?P<year>\d{{4}})[_\s-](?P<month>{_MONTH_ABBR_PAT})$",
        re.IGNORECASE,
    ),
    re.compile(
        rf"^(?P<metric>.+?)[_\s-](?P<year>\d{{4}})[_\s-](?P<month>{_MONTH_FULL_PAT})$",
        re.IGNORECASE,
    ),
    # revenue_2024 (year-only, lowest priority)
    re.compile(
        r"^(?P<metric>.+?)[_\s-](?P<year>\d{4})$",
        re.IGNORECASE,
    ),
]

# Quarter → first month of that quarter
_QUARTER_START_MONTH: dict[int, int] = {1: 1, 2: 4, 3: 7, 4: 10}

# ---------------------------------------------------------------------------
# Typed outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnTemporalParse:
    """Parsed temporal components from a single column name."""

    original_column: str
    metric_name: str
    year: int
    month: int | None = None
    quarter: int | None = None
    timestamp_iso: str = ""  # e.g. "2024-01-01"


@dataclass(frozen=True)
class TransformMeta:
    """Metadata returned alongside the normalised DataFrame."""

    preserved_rows: int
    dropped_rows: int
    parse_failures: list[str]
    inferred_periodicity: str  # "quarterly" | "monthly" | "yearly" | "mixed" | "none"
    wide_columns_detected: tuple[str, ...]
    id_columns: tuple[str, ...]
    column_parses: tuple[ColumnTemporalParse, ...] = ()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _resolve_month(raw: str) -> int | None:
    """Resolve a month name or abbreviation to 1-12, or None."""
    low = raw.strip().lower()
    if low in _MONTH_ABBR:
        return _MONTH_ABBR[low]
    if low in _MONTH_FULL:
        return _MONTH_FULL[low]
    return None


def _parse_column_temporal(column: str) -> ColumnTemporalParse | None:
    """Try each temporal pattern against *column*; return first match."""
    for pattern in _TEMPORAL_PATTERNS:
        m = pattern.match(column.strip())
        if not m:
            continue

        groups = m.groupdict()
        metric = groups["metric"].strip().strip("_- ")
        if not metric:
            continue

        year = int(groups["year"])
        quarter_str = groups.get("quarter")
        month_str = groups.get("month")

        month: int | None = None
        quarter: int | None = None

        if quarter_str is not None:
            quarter = int(quarter_str)
            month = _QUARTER_START_MONTH[quarter]
        elif month_str is not None:
            month = _resolve_month(month_str)
            if month is None:
                continue
        else:
            # Year-only
            month = 1

        ts = f"{year}-{month:02d}-01"
        return ColumnTemporalParse(
            original_column=column,
            metric_name=metric,
            year=year,
            month=month,
            quarter=quarter,
            timestamp_iso=ts,
        )

    return None


def _infer_periodicity(parses: Sequence[ColumnTemporalParse]) -> str:
    """Infer the dominant periodicity from a set of parsed columns."""
    if not parses:
        return "none"

    has_quarter = any(p.quarter is not None for p in parses)
    has_month = any(p.quarter is None and p.month is not None and p.month != 1 for p in parses)
    # year-only: month==1 and quarter is None
    has_year_only = any(
        p.quarter is None and (p.month is None or p.month == 1)
        for p in parses
        # but only if no matching month parse exists for the same year
    )

    kinds: set[str] = set()
    if has_quarter:
        kinds.add("quarterly")
    if has_month:
        kinds.add("monthly")
    if has_year_only and not has_month and not has_quarter:
        kinds.add("yearly")

    if len(kinds) == 0:
        return "none"
    if len(kinds) == 1:
        return kinds.pop()
    return "mixed"


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------


class WideToLongNormalizer:
    """
    Detects wide metric columns in a DataFrame and melts them into
    canonical long rows.

    Usage::

        normalizer = WideToLongNormalizer()
        meta = normalizer.detect(df)
        if meta.wide_columns_detected:
            long_df, transform_meta = normalizer.normalize(df)
    """

    def __init__(self, *, min_wide_columns: int = 2) -> None:
        self._min_wide_columns = max(1, min_wide_columns)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(
        self,
        df: pd.DataFrame,
    ) -> TransformMeta:
        """Analyse a DataFrame and return detection metadata (no mutation)."""
        headers = [str(c) for c in df.columns]
        parses: list[ColumnTemporalParse] = []
        failures: list[str] = []

        for header in headers:
            parsed = _parse_column_temporal(header)
            if parsed is not None:
                parses.append(parsed)

        wide_cols = tuple(p.original_column for p in parses)
        id_cols = tuple(h for h in headers if h not in set(wide_cols))

        periodicity = _infer_periodicity(parses)

        return TransformMeta(
            preserved_rows=len(df),
            dropped_rows=0,
            parse_failures=failures,
            inferred_periodicity=periodicity,
            wide_columns_detected=wide_cols,
            id_columns=id_cols,
            column_parses=tuple(parses),
        )

    # ------------------------------------------------------------------
    # Normalisation (melt)
    # ------------------------------------------------------------------

    def normalize(
        self,
        df: pd.DataFrame,
        *,
        meta: TransformMeta | None = None,
    ) -> tuple[pd.DataFrame, TransformMeta]:
        """
        Convert a wide DataFrame to long format.

        Returns:
            (long_df, metadata) — the melted DataFrame and transform metadata.
        """
        if meta is None:
            meta = self.detect(df)

        if len(meta.wide_columns_detected) < self._min_wide_columns:
            return df.copy(), TransformMeta(
                preserved_rows=len(df),
                dropped_rows=0,
                parse_failures=list(meta.parse_failures),
                inferred_periodicity="none",
                wide_columns_detected=(),
                id_columns=tuple(str(c) for c in df.columns),
                column_parses=(),
            )

        # Build lookup: column → parse
        parse_map: dict[str, ColumnTemporalParse] = {
            p.original_column: p for p in meta.column_parses
        }

        id_vars = list(meta.id_columns)
        value_vars = list(meta.wide_columns_detected)

        melted = df.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name="_wide_col_",
            value_name="metric_value",
        )

        # Map the transient _wide_col_ into metric_name + timestamp
        metric_names: list[str] = []
        timestamps: list[str] = []
        drop_mask: list[bool] = []
        parse_fail_cols: list[str] = []

        for col_name in melted["_wide_col_"]:
            parsed = parse_map.get(col_name)
            if parsed is None:
                metric_names.append(col_name)
                timestamps.append("")
                drop_mask.append(True)
                if col_name not in parse_fail_cols:
                    parse_fail_cols.append(col_name)
            else:
                metric_names.append(parsed.metric_name)
                timestamps.append(parsed.timestamp_iso)
                drop_mask.append(False)

        melted["metric_name"] = metric_names
        melted["timestamp"] = timestamps
        melted = melted.drop(columns=["_wide_col_"])

        # Drop rows where metric_value is blank/null
        before_len = len(melted)
        melted["_mv_stripped"] = melted["metric_value"].astype(str).str.strip()
        empty_mask = melted["_mv_stripped"].isin(["", "nan", "None", "NaN"])
        dropped_empty = int(empty_mask.sum())
        melted = melted[~empty_mask].drop(columns=["_mv_stripped"])

        # Drop parse-failure rows
        drop_series = pd.Series(drop_mask, index=range(before_len))
        # Align with melted after empty-drop
        drop_series = drop_series[melted.index] if len(drop_series) == before_len else pd.Series(False, index=melted.index)
        dropped_parse = 0
        if drop_series.any():
            dropped_parse = int(drop_series.sum())
            melted = melted[~drop_series]

        melted = melted.reset_index(drop=True)

        result_meta = TransformMeta(
            preserved_rows=len(melted),
            dropped_rows=dropped_empty + dropped_parse,
            parse_failures=parse_fail_cols,
            inferred_periodicity=meta.inferred_periodicity,
            wide_columns_detected=meta.wide_columns_detected,
            id_columns=meta.id_columns,
            column_parses=meta.column_parses,
        )

        logger.info(
            "Wide-to-long normalization: preserved=%d dropped=%d periodicity=%s",
            result_meta.preserved_rows,
            result_meta.dropped_rows,
            result_meta.inferred_periodicity,
        )

        return melted, result_meta
