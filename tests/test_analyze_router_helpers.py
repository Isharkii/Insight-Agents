from types import SimpleNamespace

from datetime import datetime, timezone

from app.api.routers.analyze_router import (
    _derive_kpi_period_bounds,
    _is_benign_no_valid_records,
)


def test_is_benign_no_valid_records_true_for_single_synthetic_error() -> None:
    summary = SimpleNamespace(
        rows_failed=0,
        validation_errors=[SimpleNamespace(code="no_valid_records")],
    )
    assert _is_benign_no_valid_records(summary) is True


def test_is_benign_no_valid_records_false_when_rows_failed_present() -> None:
    summary = SimpleNamespace(
        rows_failed=3,
        validation_errors=[SimpleNamespace(code="no_valid_records")],
    )
    assert _is_benign_no_valid_records(summary) is False


def test_is_benign_no_valid_records_false_for_multiple_errors() -> None:
    summary = SimpleNamespace(
        rows_failed=0,
        validation_errors=[
            SimpleNamespace(code="required_value_missing"),
            SimpleNamespace(code="no_valid_records"),
        ],
    )
    assert _is_benign_no_valid_records(summary) is False


def test_derive_kpi_period_bounds_uses_latest_data_not_wall_clock_now() -> None:
    now = datetime(2026, 2, 24, tzinfo=timezone.utc)
    earliest = datetime(2020, 1, 1, tzinfo=timezone.utc)
    latest = datetime(2024, 2, 1, tzinfo=timezone.utc)

    start, end = _derive_kpi_period_bounds(
        earliest=earliest,
        latest=latest,
        now=now,
    )

    assert end == latest
    assert start == datetime(2023, 11, 3, tzinfo=timezone.utc)


def test_derive_kpi_period_bounds_falls_back_when_no_data() -> None:
    now = datetime(2026, 2, 24, tzinfo=timezone.utc)
    start, end = _derive_kpi_period_bounds(
        earliest=None,
        latest=None,
        now=now,
    )

    assert end == now
    assert start == datetime(2025, 11, 26, tzinfo=timezone.utc)
