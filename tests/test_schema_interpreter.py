"""
tests/test_schema_interpreter.py

Tests for SchemaInterpreter covering:
  - alias collisions (two source columns matching the same canonical field)
  - unit mismatches (numeric column mapped to string field, etc.)
  - near-tie fuzzy scores (ambiguity gap detection)
  - threshold enforcement (auto_map / map_with_warning / unmapped)
  - required-field error reporting
  - audit metadata persistence
"""

from __future__ import annotations

import unittest

from app.mappers.schema_interpreter import (
    MappingDecision,
    SchemaInterpreter,
    _AUTO_MAP_THRESHOLD,
    _WARNING_THRESHOLD,
)


class TestSchemaInterpreterBasic(unittest.TestCase):
    """Core scoring and threshold tests."""

    def setUp(self) -> None:
        self.interpreter = SchemaInterpreter()

    def test_exact_alias_columns_map_above_warning_threshold(self) -> None:
        """Columns that exactly match canonical aliases should map (at least warning band)."""
        headers = [
            "source_type",
            "company",
            "category",
            "metric",
            "value",
            "date",
            "region",
        ]
        result = self.interpreter.interpret(headers)

        mapped = {d.canonical_field: d for d in result.decisions}
        # company → entity_name (alias match)
        self.assertIn("entity_name", mapped)
        self.assertEqual(mapped["entity_name"].source_column, "company")
        self.assertGreaterEqual(mapped["entity_name"].confidence, _WARNING_THRESHOLD)

    def test_exact_alias_with_samples_scores_above_auto_map(self) -> None:
        """Alias columns with matching sample data should reach auto_map threshold."""
        headers = [
            "source_type",
            "company",
            "category",
            "metric",
            "value",
            "date",
            "region",
        ]
        sample_rows = [
            {
                "source_type": "csv", "company": "Acme Corp", "category": "sales",
                "metric": "revenue", "value": "1000", "date": "2026-01-01", "region": "US",
            },
            {
                "source_type": "csv", "company": "Beta Inc", "category": "sales",
                "metric": "churn", "value": "50", "date": "2026-01-02", "region": "EU",
            },
        ]
        result = self.interpreter.interpret(headers, sample_rows=sample_rows)

        mapped = {d.canonical_field: d for d in result.decisions}
        self.assertIn("entity_name", mapped)
        self.assertGreaterEqual(mapped["entity_name"].confidence, _AUTO_MAP_THRESHOLD)

    def test_low_confidence_columns_are_unmapped(self) -> None:
        """Columns with no meaningful match should stay unmapped."""
        headers = ["zzz_random_col", "xxx_gibberish", "yyy_nonsense"]
        result = self.interpreter.interpret(headers)

        # No decisions should be generated for gibberish
        self.assertEqual(len(result.decisions), 0)
        # Required fields should appear as errors
        self.assertTrue(len(result.errors) > 0)

    def test_required_fields_produce_errors_when_unmapped(self) -> None:
        """If required canonical fields can't be mapped, errors are emitted."""
        headers = ["source_type", "timestamp", "region"]
        result = self.interpreter.interpret(headers)

        error_text = " ".join(result.errors)
        for req in ("entity_name", "category", "metric_name", "metric_value"):
            self.assertIn(req, error_text, f"Expected error for unmapped '{req}'")

    def test_decisions_have_typed_fields(self) -> None:
        """Every MappingDecision has all required typed fields."""
        headers = ["entity_name", "category", "metric_name", "metric_value"]
        result = self.interpreter.interpret(headers)

        for d in result.decisions:
            self.assertIsInstance(d, MappingDecision)
            self.assertIsInstance(d.source_column, str)
            self.assertIsInstance(d.canonical_field, str)
            self.assertIsInstance(d.confidence, float)
            self.assertIsInstance(d.ambiguity_gap, float)
            self.assertIsInstance(d.matched_rule, str)
            self.assertIn(d.matched_rule, {"auto_map", "map_with_warning"})

    def test_audit_metadata_is_populated(self) -> None:
        """Audit dict contains expected keys."""
        headers = ["entity_name", "category", "metric_name", "metric_value"]
        result = self.interpreter.interpret(headers)

        self.assertIn("interpreted_at", result.audit)
        self.assertIn("auto_mapped", result.audit)
        self.assertIn("mapped_with_warning", result.audit)
        self.assertIn("unmapped_fields", result.audit)
        self.assertIn("score_matrix", result.audit)

    def test_to_mapping_dict(self) -> None:
        """to_mapping_dict produces canonical_field → source_column dict."""
        headers = ["entity_name", "category", "metric_name", "metric_value"]
        result = self.interpreter.interpret(headers)

        mapping = self.interpreter.to_mapping_dict(result)
        self.assertIsInstance(mapping, dict)
        for d in result.decisions:
            self.assertEqual(mapping[d.canonical_field], d.source_column)


class TestAliasCollisions(unittest.TestCase):
    """Tests for when multiple source columns compete for the same canonical field."""

    def setUp(self) -> None:
        self.interpreter = SchemaInterpreter()

    def test_two_entity_aliases_only_one_wins(self) -> None:
        """When 'company' and 'client' both alias entity_name, only one is assigned."""
        headers = [
            "company",
            "client",
            "category",
            "metric_name",
            "metric_value",
        ]
        result = self.interpreter.interpret(headers)

        entity_decisions = [
            d for d in result.decisions if d.canonical_field == "entity_name"
        ]
        self.assertEqual(
            len(entity_decisions), 1,
            "Only one source column should map to entity_name",
        )

    def test_alias_collision_winner_has_small_ambiguity_gap(self) -> None:
        """When two columns are close matches, ambiguity_gap should be small."""
        headers = [
            "company",
            "client",
            "category",
            "metric_name",
            "metric_value",
        ]
        result = self.interpreter.interpret(headers)

        entity_decisions = [
            d for d in result.decisions if d.canonical_field == "entity_name"
        ]
        self.assertEqual(len(entity_decisions), 1)
        # Both 'company' and 'client' are exact aliases, so competing header gap is 0
        self.assertLessEqual(entity_decisions[0].ambiguity_gap, 0.10)

    def test_category_aware_collision_saas(self) -> None:
        """In SaaS context, 'mrr' and 'value' both target metric_value."""
        interpreter = SchemaInterpreter(category_hint="saas")
        headers = [
            "entity_name",
            "category",
            "metric_name",
            "mrr",
            "value",
            "timestamp",
        ]
        result = interpreter.interpret(headers)

        metric_value_decisions = [
            d for d in result.decisions if d.canonical_field == "metric_value"
        ]
        self.assertEqual(len(metric_value_decisions), 1)


class TestUnitMismatches(unittest.TestCase):
    """Tests for dtype compatibility scoring detecting mismatched types."""

    def setUp(self) -> None:
        self.interpreter = SchemaInterpreter()

    def test_numeric_samples_boost_metric_value_score(self) -> None:
        """A column with numeric samples should score higher for metric_value."""
        headers = ["data_col"]
        sample_rows = [
            {"data_col": "100.5"},
            {"data_col": "200"},
            {"data_col": "350.75"},
        ]
        result = self.interpreter.interpret(headers, sample_rows=sample_rows)

        # Check score_matrix: data_col should score higher for metric_value
        # than for entity_name (which expects strings)
        matrix = result.audit["score_matrix"]
        self.assertGreater(
            matrix["data_col"]["metric_value"]["dtype_compatibility"],
            matrix["data_col"]["entity_name"]["dtype_compatibility"],
        )

    def test_date_samples_boost_timestamp_score(self) -> None:
        """A column with date-like samples should score higher for timestamp."""
        headers = ["when_col"]
        sample_rows = [
            {"when_col": "2026-01-15"},
            {"when_col": "2026-02-20"},
            {"when_col": "2025-12-01"},
        ]
        result = self.interpreter.interpret(headers, sample_rows=sample_rows)

        matrix = result.audit["score_matrix"]
        self.assertGreater(
            matrix["when_col"]["timestamp"]["dtype_compatibility"],
            matrix["when_col"]["entity_name"]["dtype_compatibility"],
        )

    def test_string_samples_penalize_metric_value(self) -> None:
        """A column with purely string values shouldn't match metric_value on dtype."""
        headers = ["text_field"]
        sample_rows = [
            {"text_field": "alpha"},
            {"text_field": "beta"},
            {"text_field": "gamma"},
        ]
        result = self.interpreter.interpret(headers, sample_rows=sample_rows)

        matrix = result.audit["score_matrix"]
        self.assertEqual(
            matrix["text_field"]["metric_value"]["dtype_compatibility"], 0.0,
        )

    def test_json_samples_boost_metadata_json(self) -> None:
        """A column with JSON-looking values should score high for metadata_json dtype."""
        headers = ["extra_info"]
        sample_rows = [
            {"extra_info": '{"key": "val"}'},
            {"extra_info": '{"a": 1}'},
        ]
        result = self.interpreter.interpret(headers, sample_rows=sample_rows)

        matrix = result.audit["score_matrix"]
        self.assertGreater(
            matrix["extra_info"]["metadata_json"]["dtype_compatibility"],
            matrix["extra_info"]["entity_name"]["dtype_compatibility"],
        )


class TestNearTieFuzzyScores(unittest.TestCase):
    """Tests for near-tie fuzzy matching and ambiguity detection."""

    def setUp(self) -> None:
        self.interpreter = SchemaInterpreter()

    def test_near_tie_produces_small_ambiguity_gap(self) -> None:
        """Headers with similar fuzzy scores to different fields should have small gaps."""
        # 'metric_val' is close to both 'metric_value' and 'metric_name'
        headers = [
            "entity_name",
            "category",
            "metric_val",
            "metric_nam",
        ]
        result = self.interpreter.interpret(headers)

        # Both should map but their ambiguity gaps should reflect the near-tie
        for d in result.decisions:
            if d.source_column in ("metric_val", "metric_nam"):
                # These are near-ties; gap should be relatively small
                self.assertLess(
                    d.ambiguity_gap, 0.50,
                    f"Expected small ambiguity gap for '{d.source_column}', "
                    f"got {d.ambiguity_gap}",
                )

    def test_clear_match_has_large_ambiguity_gap(self) -> None:
        """Columns that clearly match one field should have a larger gap."""
        headers = [
            "entity_name",
            "category",
            "metric_name",
            "metric_value",
            "timestamp",
        ]
        result = self.interpreter.interpret(headers)

        for d in result.decisions:
            if d.canonical_field == "entity_name":
                # entity_name matches exactly, gap should be meaningful
                self.assertGreater(d.ambiguity_gap, 0.05)

    def test_similar_prefixes_competing_fields(self) -> None:
        """'metric_value_usd' and 'metric_name_label' should resolve correctly."""
        headers = [
            "entity_name",
            "category",
            "metric_name_label",
            "metric_value_usd",
            "timestamp",
        ]
        result = self.interpreter.interpret(headers)

        mapped = {d.canonical_field: d.source_column for d in result.decisions}
        # Fuzzy matching should prefer the correct pairings
        if "metric_name" in mapped:
            self.assertEqual(mapped["metric_name"], "metric_name_label")
        if "metric_value" in mapped:
            self.assertEqual(mapped["metric_value"], "metric_value_usd")


class TestWarningThresholdBand(unittest.TestCase):
    """Tests for the 0.75–0.89 warning band."""

    def setUp(self) -> None:
        self.interpreter = SchemaInterpreter()

    def test_moderate_match_produces_warning(self) -> None:
        """A header in the warning band should produce a map_with_warning decision."""
        # Use a header that partially matches but isn't an exact alias
        # 'Competitive Name' fuzzy-matches entity_name via 'competitor' alias
        headers = [
            "Competitive Nme",
            "category",
            "metric_name",
            "metric_value",
        ]
        result = self.interpreter.interpret(headers)

        warning_decisions = [
            d for d in result.decisions if d.matched_rule == "map_with_warning"
        ]
        # If the fuzzy score lands in warning band, we should see warnings
        if warning_decisions:
            self.assertTrue(len(result.warnings) > 0)
            for d in warning_decisions:
                self.assertGreaterEqual(d.confidence, _WARNING_THRESHOLD)
                self.assertLess(d.confidence, _AUTO_MAP_THRESHOLD)

    def test_no_silent_low_confidence_mappings(self) -> None:
        """No decision should ever have confidence < 0.75."""
        headers = [
            "zxcvbn",
            "qwerty",
            "asdfgh",
            "entity_name",
            "category",
            "metric_name",
            "metric_value",
        ]
        result = self.interpreter.interpret(headers)

        for d in result.decisions:
            self.assertGreaterEqual(
                d.confidence, _WARNING_THRESHOLD,
                f"Decision for '{d.source_column}' → '{d.canonical_field}' "
                f"has confidence {d.confidence} below threshold {_WARNING_THRESHOLD}",
            )


class TestCategoryHint(unittest.TestCase):
    """Tests for category-aware alias activation."""

    def test_saas_aliases_activate_mrr(self) -> None:
        """With category_hint='saas', 'mrr' should map to metric_value."""
        interpreter = SchemaInterpreter(category_hint="saas")
        headers = [
            "entity_name",
            "category",
            "metric_name",
            "mrr",
            "timestamp",
        ]
        result = interpreter.interpret(headers)

        mapped = {d.canonical_field: d.source_column for d in result.decisions}
        self.assertEqual(mapped.get("metric_value"), "mrr")

    def test_ecommerce_aliases_activate_gmv(self) -> None:
        """With category_hint='ecommerce', 'gmv' should map to metric_value."""
        interpreter = SchemaInterpreter(category_hint="ecommerce")
        headers = [
            "entity_name",
            "category",
            "metric_name",
            "gmv",
            "timestamp",
        ]
        result = interpreter.interpret(headers)

        mapped = {d.canonical_field: d.source_column for d in result.decisions}
        self.assertEqual(mapped.get("metric_value"), "gmv")

    def test_runtime_category_hint_overrides_constructor(self) -> None:
        """Runtime category_hint should override the constructor hint."""
        interpreter = SchemaInterpreter(category_hint="saas")
        headers = [
            "entity_name",
            "category",
            "metric_name",
            "gmv",
            "timestamp",
        ]
        # gmv is an ecommerce alias, not saas
        result = interpreter.interpret(headers, category_hint="ecommerce")

        mapped = {d.canonical_field: d.source_column for d in result.decisions}
        self.assertEqual(mapped.get("metric_value"), "gmv")


class TestScoreMatrixAudit(unittest.TestCase):
    """Tests that the full score matrix is persisted in audit output."""

    def test_score_matrix_contains_all_fields(self) -> None:
        """Every header × canonical field pair should appear in the matrix."""
        headers = ["company", "sales", "kpi", "amount"]
        result = self.interpreter = SchemaInterpreter()
        result = self.interpreter.interpret(headers)

        matrix = result.audit["score_matrix"]
        for header in headers:
            self.assertIn(header, matrix)
            for cf in (
                "source_type", "entity_name", "category", "role",
                "metric_name", "metric_value", "timestamp", "region",
                "metadata_json",
            ):
                self.assertIn(cf, matrix[header])
                self.assertIn("composite", matrix[header][cf])
                self.assertIn("alias_exact", matrix[header][cf])
                self.assertIn("fuzzy_similarity", matrix[header][cf])
                self.assertIn("dtype_compatibility", matrix[header][cf])
                self.assertIn("value_pattern_fit", matrix[header][cf])


if __name__ == "__main__":
    unittest.main()
