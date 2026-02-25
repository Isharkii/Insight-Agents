"""
tests/test_wide_to_long_normalizer.py

Tests for WideToLongNormalizer covering:
  - quarterly wide schemas  (revenue_2024_Q1 … Q4)
  - monthly wide schemas    (churn_Jan_2025 … Dec_2025)
  - mixed wide schemas      (quarterly + monthly + year-only in one file)
  - id-column preservation  (entity, category survive the melt)
  - empty / blank value pruning
  - transform metadata accuracy
  - long-format passthrough  (no-op when no wide columns detected)
"""

from __future__ import annotations

import unittest

import pandas as pd

from app.mappers.wide_to_long_normalizer import (
    ColumnTemporalParse,
    TransformMeta,
    WideToLongNormalizer,
    _parse_column_temporal,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _quarterly_df() -> pd.DataFrame:
    """Two entities × four quarters of revenue."""
    return pd.DataFrame({
        "entity_name": ["Acme Corp", "Beta Inc"],
        "category":    ["sales", "sales"],
        "revenue_2024_Q1": ["1000", "2000"],
        "revenue_2024_Q2": ["1100", "2100"],
        "revenue_2024_Q3": ["1200", "2200"],
        "revenue_2024_Q4": ["1300", "2300"],
    })


def _monthly_df() -> pd.DataFrame:
    """One entity × three months of churn."""
    return pd.DataFrame({
        "entity_name": ["Acme Corp", "Beta Inc"],
        "category":    ["saas", "saas"],
        "churn_Jan_2025": ["5", "8"],
        "churn_Feb_2025": ["4", "7"],
        "churn_Mar_2025": ["6", "9"],
    })


def _monthly_full_name_df() -> pd.DataFrame:
    """Full month names: churn_January_2025."""
    return pd.DataFrame({
        "entity_name": ["Acme Corp"],
        "category":    ["saas"],
        "churn_January_2025": ["5"],
        "churn_February_2025": ["4"],
    })


def _mixed_df() -> pd.DataFrame:
    """Mixed periodicity: quarterly revenue + monthly churn + yearly cost."""
    return pd.DataFrame({
        "entity_name": ["Acme Corp"],
        "category":    ["sales"],
        "revenue_2024_Q1": ["1000"],
        "revenue_2024_Q2": ["1100"],
        "churn_Jan_2025":  ["5"],
        "churn_Feb_2025":  ["4"],
        "cost_2024":       ["5000"],
    })


def _long_format_df() -> pd.DataFrame:
    """Already long — no wide columns."""
    return pd.DataFrame({
        "entity_name":  ["Acme Corp", "Beta Inc"],
        "category":     ["sales", "sales"],
        "metric_name":  ["revenue", "revenue"],
        "metric_value": ["1000", "2000"],
        "timestamp":    ["2024-01-01", "2024-01-01"],
    })


def _wide_with_blanks_df() -> pd.DataFrame:
    """Wide columns where some cells are empty."""
    return pd.DataFrame({
        "entity_name": ["Acme Corp", "Beta Inc"],
        "category":    ["sales", "sales"],
        "revenue_2024_Q1": ["1000", ""],
        "revenue_2024_Q2": ["", ""],
    })


def _reverse_quarter_df() -> pd.DataFrame:
    """Reverse order: revenue_Q1_2024."""
    return pd.DataFrame({
        "entity_name": ["Acme Corp"],
        "category":    ["sales"],
        "revenue_Q1_2024": ["1000"],
        "revenue_Q2_2024": ["1100"],
    })


def _reverse_month_df() -> pd.DataFrame:
    """Reverse order: churn_2025_Jan."""
    return pd.DataFrame({
        "entity_name": ["Acme Corp"],
        "category":    ["saas"],
        "churn_2025_Jan": ["5"],
        "churn_2025_Feb": ["4"],
    })


# ===================================================================
# Column temporal parsing
# ===================================================================


class TestColumnTemporalParsing(unittest.TestCase):
    """Unit tests for _parse_column_temporal."""

    def test_quarterly_standard(self) -> None:
        p = _parse_column_temporal("revenue_2024_Q1")
        assert p is not None
        self.assertEqual(p.metric_name, "revenue")
        self.assertEqual(p.year, 2024)
        self.assertEqual(p.quarter, 1)
        self.assertEqual(p.timestamp_iso, "2024-01-01")

    def test_quarterly_q4(self) -> None:
        p = _parse_column_temporal("churn_2024_Q4")
        assert p is not None
        self.assertEqual(p.quarter, 4)
        self.assertEqual(p.timestamp_iso, "2024-10-01")

    def test_quarterly_reverse(self) -> None:
        p = _parse_column_temporal("revenue_Q2_2024")
        assert p is not None
        self.assertEqual(p.metric_name, "revenue")
        self.assertEqual(p.quarter, 2)
        self.assertEqual(p.timestamp_iso, "2024-04-01")

    def test_monthly_abbreviation(self) -> None:
        p = _parse_column_temporal("churn_Jan_2025")
        assert p is not None
        self.assertEqual(p.metric_name, "churn")
        self.assertEqual(p.month, 1)
        self.assertEqual(p.year, 2025)
        self.assertEqual(p.timestamp_iso, "2025-01-01")

    def test_monthly_full_name(self) -> None:
        p = _parse_column_temporal("churn_February_2025")
        assert p is not None
        self.assertEqual(p.metric_name, "churn")
        self.assertEqual(p.month, 2)
        self.assertEqual(p.timestamp_iso, "2025-02-01")

    def test_monthly_reverse(self) -> None:
        p = _parse_column_temporal("churn_2025_Mar")
        assert p is not None
        self.assertEqual(p.metric_name, "churn")
        self.assertEqual(p.month, 3)
        self.assertEqual(p.year, 2025)

    def test_yearly(self) -> None:
        p = _parse_column_temporal("cost_2024")
        assert p is not None
        self.assertEqual(p.metric_name, "cost")
        self.assertEqual(p.year, 2024)
        self.assertEqual(p.timestamp_iso, "2024-01-01")

    def test_no_match(self) -> None:
        self.assertIsNone(_parse_column_temporal("entity_name"))
        self.assertIsNone(_parse_column_temporal("category"))
        self.assertIsNone(_parse_column_temporal("metric_value"))

    def test_hyphen_separator(self) -> None:
        p = _parse_column_temporal("revenue-2024-Q1")
        assert p is not None
        self.assertEqual(p.metric_name, "revenue")
        self.assertEqual(p.quarter, 1)

    def test_space_separator(self) -> None:
        p = _parse_column_temporal("revenue 2024 Q3")
        assert p is not None
        self.assertEqual(p.quarter, 3)

    def test_case_insensitive_quarter(self) -> None:
        p = _parse_column_temporal("revenue_2024_q2")
        assert p is not None
        self.assertEqual(p.quarter, 2)

    def test_multi_word_metric(self) -> None:
        p = _parse_column_temporal("net_revenue_2024_Q1")
        assert p is not None
        self.assertEqual(p.metric_name, "net_revenue")


# ===================================================================
# Quarterly fixtures
# ===================================================================


class TestQuarterlySchema(unittest.TestCase):
    """Tests with quarterly wide-format data."""

    def setUp(self) -> None:
        self.normalizer = WideToLongNormalizer()
        self.df = _quarterly_df()

    def test_detects_four_wide_columns(self) -> None:
        meta = self.normalizer.detect(self.df)
        self.assertEqual(len(meta.wide_columns_detected), 4)
        self.assertEqual(meta.inferred_periodicity, "quarterly")

    def test_id_columns_preserved(self) -> None:
        meta = self.normalizer.detect(self.df)
        self.assertIn("entity_name", meta.id_columns)
        self.assertIn("category", meta.id_columns)

    def test_normalize_produces_long_rows(self) -> None:
        long_df, meta = self.normalizer.normalize(self.df)
        # 2 entities × 4 quarters = 8 rows
        self.assertEqual(len(long_df), 8)
        self.assertEqual(meta.preserved_rows, 8)
        self.assertEqual(meta.dropped_rows, 0)

    def test_normalize_has_canonical_columns(self) -> None:
        long_df, _ = self.normalizer.normalize(self.df)
        self.assertIn("entity_name", long_df.columns)
        self.assertIn("category", long_df.columns)
        self.assertIn("metric_name", long_df.columns)
        self.assertIn("metric_value", long_df.columns)
        self.assertIn("timestamp", long_df.columns)

    def test_metric_names_are_cleaned(self) -> None:
        long_df, _ = self.normalizer.normalize(self.df)
        unique_metrics = set(long_df["metric_name"].unique())
        self.assertEqual(unique_metrics, {"revenue"})

    def test_timestamps_are_quarter_starts(self) -> None:
        long_df, _ = self.normalizer.normalize(self.df)
        ts_values = sorted(long_df["timestamp"].unique())
        self.assertEqual(ts_values, [
            "2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01",
        ])

    def test_entity_values_preserved(self) -> None:
        long_df, _ = self.normalizer.normalize(self.df)
        entities = sorted(long_df["entity_name"].unique())
        self.assertEqual(entities, ["Acme Corp", "Beta Inc"])

    def test_reverse_quarter_format(self) -> None:
        long_df, meta = self.normalizer.normalize(_reverse_quarter_df())
        self.assertEqual(len(long_df), 2)
        self.assertEqual(meta.inferred_periodicity, "quarterly")
        ts_values = sorted(long_df["timestamp"].unique())
        self.assertEqual(ts_values, ["2024-01-01", "2024-04-01"])


# ===================================================================
# Monthly fixtures
# ===================================================================


class TestMonthlySchema(unittest.TestCase):
    """Tests with monthly wide-format data."""

    def setUp(self) -> None:
        self.normalizer = WideToLongNormalizer()

    def test_detects_monthly_columns(self) -> None:
        meta = self.normalizer.detect(_monthly_df())
        self.assertEqual(len(meta.wide_columns_detected), 3)
        self.assertEqual(meta.inferred_periodicity, "monthly")

    def test_normalize_monthly_abbreviation(self) -> None:
        long_df, meta = self.normalizer.normalize(_monthly_df())
        # 2 entities × 3 months = 6 rows
        self.assertEqual(len(long_df), 6)
        unique_metrics = set(long_df["metric_name"].unique())
        self.assertEqual(unique_metrics, {"churn"})
        ts_values = sorted(long_df["timestamp"].unique())
        self.assertEqual(ts_values, ["2025-01-01", "2025-02-01", "2025-03-01"])

    def test_normalize_monthly_full_name(self) -> None:
        long_df, meta = self.normalizer.normalize(_monthly_full_name_df())
        self.assertEqual(len(long_df), 2)
        self.assertEqual(meta.inferred_periodicity, "monthly")

    def test_reverse_month_format(self) -> None:
        long_df, meta = self.normalizer.normalize(_reverse_month_df())
        self.assertEqual(len(long_df), 2)
        ts_values = sorted(long_df["timestamp"].unique())
        self.assertEqual(ts_values, ["2025-01-01", "2025-02-01"])


# ===================================================================
# Mixed wide schemas
# ===================================================================


class TestMixedWideSchema(unittest.TestCase):
    """Tests with mixed periodicity (quarterly + monthly + yearly)."""

    def setUp(self) -> None:
        self.normalizer = WideToLongNormalizer()
        self.df = _mixed_df()

    def test_detects_mixed_periodicity(self) -> None:
        meta = self.normalizer.detect(self.df)
        self.assertEqual(meta.inferred_periodicity, "mixed")
        self.assertEqual(len(meta.wide_columns_detected), 5)

    def test_normalize_mixed_produces_correct_rows(self) -> None:
        long_df, meta = self.normalizer.normalize(self.df)
        # 1 entity × 5 wide columns = 5 rows
        self.assertEqual(len(long_df), 5)
        self.assertEqual(meta.preserved_rows, 5)

    def test_normalize_mixed_extracts_distinct_metrics(self) -> None:
        long_df, _ = self.normalizer.normalize(self.df)
        metrics = sorted(long_df["metric_name"].unique())
        self.assertEqual(metrics, ["churn", "cost", "revenue"])

    def test_normalize_mixed_timestamps(self) -> None:
        long_df, _ = self.normalizer.normalize(self.df)
        ts = sorted(long_df["timestamp"].unique())
        # Q1=Jan, Q2=Apr, Jan, Feb, 2024=Jan
        self.assertIn("2024-01-01", ts)
        self.assertIn("2024-04-01", ts)
        self.assertIn("2025-01-01", ts)
        self.assertIn("2025-02-01", ts)


# ===================================================================
# Long-format passthrough
# ===================================================================


class TestLongFormatPassthrough(unittest.TestCase):
    """Already long-format data should pass through unchanged."""

    def setUp(self) -> None:
        self.normalizer = WideToLongNormalizer()

    def test_no_wide_columns_detected(self) -> None:
        meta = self.normalizer.detect(_long_format_df())
        self.assertEqual(len(meta.wide_columns_detected), 0)
        self.assertEqual(meta.inferred_periodicity, "none")

    def test_normalize_returns_copy_unchanged(self) -> None:
        df = _long_format_df()
        result_df, meta = self.normalizer.normalize(df)
        self.assertEqual(len(result_df), len(df))
        self.assertEqual(list(result_df.columns), list(df.columns))
        self.assertEqual(meta.preserved_rows, len(df))
        self.assertEqual(meta.dropped_rows, 0)


# ===================================================================
# Empty / blank value handling
# ===================================================================


class TestBlankValuePruning(unittest.TestCase):
    """Wide cells with blank / empty values should be dropped."""

    def setUp(self) -> None:
        self.normalizer = WideToLongNormalizer()

    def test_blank_cells_are_dropped(self) -> None:
        long_df, meta = self.normalizer.normalize(_wide_with_blanks_df())
        # 2 entities × 2 quarters = 4 potential rows
        # But 3 cells are blank → only 1 row preserved
        self.assertEqual(meta.preserved_rows, 1)
        self.assertEqual(meta.dropped_rows, 3)
        self.assertEqual(long_df.iloc[0]["metric_value"], "1000")


# ===================================================================
# Transform metadata accuracy
# ===================================================================


class TestTransformMeta(unittest.TestCase):
    """TransformMeta correctness."""

    def setUp(self) -> None:
        self.normalizer = WideToLongNormalizer()

    def test_meta_contains_column_parses(self) -> None:
        meta = self.normalizer.detect(_quarterly_df())
        self.assertEqual(len(meta.column_parses), 4)
        for cp in meta.column_parses:
            self.assertIsInstance(cp, ColumnTemporalParse)
            self.assertTrue(cp.timestamp_iso)

    def test_meta_id_columns_exclude_wide(self) -> None:
        meta = self.normalizer.detect(_quarterly_df())
        for wc in meta.wide_columns_detected:
            self.assertNotIn(wc, meta.id_columns)

    def test_meta_periodicity_quarterly(self) -> None:
        meta = self.normalizer.detect(_quarterly_df())
        self.assertEqual(meta.inferred_periodicity, "quarterly")

    def test_meta_periodicity_monthly(self) -> None:
        meta = self.normalizer.detect(_monthly_df())
        self.assertEqual(meta.inferred_periodicity, "monthly")

    def test_meta_periodicity_none_for_long(self) -> None:
        meta = self.normalizer.detect(_long_format_df())
        self.assertEqual(meta.inferred_periodicity, "none")


# ===================================================================
# Min wide columns threshold
# ===================================================================


class TestMinWideColumnsThreshold(unittest.TestCase):
    """Normalizer requires at least min_wide_columns to trigger."""

    def test_single_wide_column_is_passthrough(self) -> None:
        df = pd.DataFrame({
            "entity_name": ["Acme"],
            "category": ["sales"],
            "revenue_2024_Q1": ["1000"],
        })
        normalizer = WideToLongNormalizer(min_wide_columns=2)
        result_df, meta = normalizer.normalize(df)
        # Should NOT melt — only 1 wide column < threshold
        self.assertEqual(len(meta.wide_columns_detected), 0)
        self.assertEqual(list(result_df.columns), list(df.columns))

    def test_custom_threshold_respected(self) -> None:
        df = _quarterly_df()  # 4 wide columns
        normalizer = WideToLongNormalizer(min_wide_columns=5)
        result_df, meta = normalizer.normalize(df)
        # 4 < 5 → passthrough
        self.assertEqual(len(meta.wide_columns_detected), 0)


if __name__ == "__main__":
    unittest.main()
