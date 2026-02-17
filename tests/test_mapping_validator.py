from __future__ import annotations

import unittest

from app.validators.mapping_validator import MappingErrorDetail, MappingValidator, SchemaMappingError


class TestMappingValidator(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = MappingValidator(
            required_fields=(
                "source_type",
                "entity_name",
                "category",
                "metric_name",
                "metric_value",
                "timestamp",
            ),
            canonical_fields=(
                "source_type",
                "entity_name",
                "category",
                "metric_name",
                "metric_value",
                "timestamp",
                "region",
                "metadata_json",
            ),
        )

    def test_raises_on_missing_required_fields(self) -> None:
        with self.assertRaises(SchemaMappingError) as ctx:
            self.validator.validate(
                mapping={"source_type": "source", "entity_name": "company"},
                source_headers=("source", "company"),
            )

        codes = {error.code for error in ctx.exception.errors}
        self.assertIn("required_field_unmapped", codes)

    def test_raises_on_invalid_source_column(self) -> None:
        with self.assertRaises(SchemaMappingError) as ctx:
            self.validator.validate(
                mapping={
                    "source_type": "source",
                    "entity_name": "company",
                    "category": "category",
                    "metric_name": "metric",
                    "metric_value": "value",
                    "timestamp": "missing_column",
                },
                source_headers=("source", "company", "category", "metric", "value"),
                pre_errors=[
                    MappingErrorDetail(
                        code="override_source_not_found",
                        message="manual override missing header",
                        canonical_field="timestamp",
                        source_column="missing_column",
                    )
                ],
            )

        codes = [error.code for error in ctx.exception.errors]
        self.assertIn("unknown_source_column", codes)
        self.assertIn("override_source_not_found", codes)


if __name__ == "__main__":
    unittest.main()
