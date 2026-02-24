from __future__ import annotations

import pandas as pd

from app.services.csv_ingestion_service import CSVIngestionService


def _service() -> CSVIngestionService:
    return CSVIngestionService(
        batch_size=1000,
        max_validation_errors=100,
        log_validation_errors=False,
    )


def test_timestamp_required_for_saas_category() -> None:
    service = _service()
    chunk = pd.DataFrame(
        [
            {
                "source_type": "csv",
                "entity_name": "Acme",
                "category": "saas",
                "metric_name": "mrr",
                "metric_value": "1000",
                "timestamp": "",
                "region": "",
                "metadata_json": "",
            }
        ]
    )

    errors = []
    valid_inputs, failed = service._validate_and_normalize_chunk(chunk, errors)  # noqa: SLF001

    assert failed == 1
    assert valid_inputs == []
    assert errors
    assert errors[0].code == "timestamp_required_for_category"


def test_timestamp_optional_for_static_dataset_category() -> None:
    service = _service()
    chunk = pd.DataFrame(
        [
            {
                "source_type": "csv",
                "entity_name": "Acme",
                "category": "static_dataset",
                "metric_name": "industry",
                "metric_value": "software",
                "timestamp": "",
                "region": "",
                "metadata_json": "",
            }
        ]
    )

    errors = []
    valid_inputs, failed = service._validate_and_normalize_chunk(chunk, errors)  # noqa: SLF001

    assert failed == 0
    assert len(valid_inputs) == 1
    assert errors == []
    assert valid_inputs[0].timestamp is not None
