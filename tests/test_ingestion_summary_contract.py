from __future__ import annotations

from app.domain.canonical_insight import IngestionSummary


def test_ingestion_summary_includes_pipeline_and_provenance_fields() -> None:
    summary = IngestionSummary(
        rows_processed=10,
        rows_failed=2,
        pipeline_status="partial",
        confidence_score=0.82,
        warnings=["2 rows failed validation"],
        provenance={"mapping_config_id": "cfg-1"},
        diagnostics={"rows_total_considered": 12},
        validation_errors=[],
    )

    assert summary.pipeline_status == "partial"
    assert summary.confidence_score == 0.82
    assert summary.provenance["mapping_config_id"] == "cfg-1"
    assert summary.diagnostics["rows_total_considered"] == 12
