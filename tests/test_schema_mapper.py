from __future__ import annotations

import unittest

from app.mappers.schema_mapper import SchemaMapper
from app.validators.csv_validator import CSVRowValidator
from app.validators.mapping_validator import SchemaMappingError
from db.models.canonical_insight_record import CanonicalInsightRecord
from db.models.mapping_config import MappingConfig


class TestSchemaMapper(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = SchemaMapper()

    def test_auto_detects_columns_with_fuzzy_matching(self) -> None:
        headers = [
            "Source Typ",
            "Competitor Nmae",
            "Insight Catgory",
            "KPI",
            "Metric Amnt",
            "Recorded At",
            "Geo Region",
        ]

        resolution = self.mapper.resolve_mapping(headers)

        self.assertEqual(resolution.canonical_to_source["source_type"], "Source Typ")
        self.assertEqual(resolution.canonical_to_source["entity_name"], "Competitor Nmae")
        self.assertEqual(resolution.canonical_to_source["category"], "Insight Catgory")
        self.assertEqual(resolution.canonical_to_source["metric_name"], "KPI")
        self.assertEqual(resolution.canonical_to_source["metric_value"], "Metric Amnt")
        self.assertEqual(resolution.canonical_to_source["timestamp"], "Recorded At")

    def test_manual_override_mapping_takes_precedence(self) -> None:
        headers = ["src", "biz", "cat", "label", "amt", "when"]
        manual = {
            "source_type": "src",
            "entity_name": "biz",
            "category": "cat",
            "metric_name": "label",
            "metric_value": "amt",
            "timestamp": "when",
        }

        resolution = self.mapper.resolve_mapping(headers, manual_overrides=manual)

        self.assertEqual(resolution.canonical_to_source["metric_name"], "label")
        self.assertEqual(resolution.match_strategies["metric_name"], "override")

    def test_invalid_manual_override_raises_structured_error(self) -> None:
        headers = [
            "source_type",
            "entity_name",
            "category",
            "metric_name",
            "metric_value",
            "timestamp",
        ]
        manual = {"unknown_field": "source_type"}

        with self.assertRaises(SchemaMappingError) as ctx:
            self.mapper.resolve_mapping(headers, manual_overrides=manual)

        error_codes = {error.code for error in ctx.exception.errors}
        self.assertIn("invalid_override_field", error_codes)

    def test_missing_required_mapping_raises_structured_error(self) -> None:
        headers = [
            "source_type",
            "entity_name",
            "category",
            "metric_name",
            "metric_value",
        ]

        with self.assertRaises(SchemaMappingError) as ctx:
            self.mapper.resolve_mapping(headers)

        missing = [
            error.canonical_field
            for error in ctx.exception.errors
            if error.code == "required_field_unmapped"
        ]
        self.assertIn("timestamp", missing)

    def test_uses_db_mapping_config_overrides(self) -> None:
        headers = ["src", "company", "kind", "signal", "value_col", "captured_on"]
        mapping_config = MappingConfig(
            name="client_a_schema",
            client_name="client_a",
            field_mapping_json={
                "source_type": "src",
                "entity_name": "company",
                "category": "kind",
                "metric_name": "signal",
                "metric_value": "value_col",
                "timestamp": "captured_on",
            },
            alias_overrides_json=None,
            is_active=True,
        )

        resolution = self.mapper.resolve_mapping(headers, mapping_config=mapping_config)

        self.assertEqual(resolution.canonical_to_source["metric_name"], "signal")
        self.assertEqual(resolution.match_strategies["metric_name"], "override")

    def test_outputs_mapped_canonical_record_object(self) -> None:
        headers = [
            "source_type",
            "entity_name",
            "category",
            "metric_name",
            "metric_value",
            "timestamp",
            "region",
            "metadata_json",
        ]
        row = {
            "source_type": "csv",
            "entity_name": "Acme",
            "category": "pricing",
            "metric_name": "monthly_price",
            "metric_value": "99.5",
            "timestamp": "2026-02-10T00:00:00Z",
            "region": "US",
            "metadata_json": "{\"source\":\"upload\"}",
        }

        resolution = self.mapper.resolve_mapping(headers)
        mapped_row = self.mapper.map_row(raw_row=row, mapping=resolution)

        parsed, errors = CSVRowValidator().validate_mapped_row(mapped_row=mapped_row, row_number=2)
        self.assertEqual(errors, [])
        assert parsed is not None

        record = self.mapper.to_canonical_record(parsed)

        self.assertIsInstance(record, CanonicalInsightRecord)
        self.assertEqual(record.entity_name, "Acme")
        self.assertEqual(record.category, "pricing")
        self.assertEqual(record.metric_value, 99.5)


if __name__ == "__main__":
    unittest.main()
