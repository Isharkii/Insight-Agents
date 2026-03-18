"""
Tests for canonical validation alias resolution, case-insensitive matching,
and diagnostic output.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.services.aggregation_service import _as_tuple
from app.services.canonical_validation import (
    CanonicalValidationResult,
    validate_canonical_inputs_for_kpi,
)
from app.services.kpi_canonical_schema import metric_aliases_for_business_type


# ---------------------------------------------------------------------------
# _as_tuple normalization
# ---------------------------------------------------------------------------


def test_as_tuple_lowercases_single_string() -> None:
    result = _as_tuple("MRR", fallback=("fallback",))
    assert result == ("mrr",)


def test_as_tuple_lowercases_sequence() -> None:
    result = _as_tuple(["Recurring_Revenue", "MRR"], fallback=("fallback",))
    assert result == ("recurring_revenue", "mrr")


def test_as_tuple_strips_whitespace() -> None:
    result = _as_tuple(["  MRR  ", " churn_rate "], fallback=("fallback",))
    assert result == ("mrr", "churn_rate")


def test_as_tuple_empty_falls_back() -> None:
    result = _as_tuple("  ", fallback=("default",))
    assert result == ("default",)


def test_as_tuple_empty_list_falls_back() -> None:
    result = _as_tuple([], fallback=("default",))
    assert result == ("default",)


# ---------------------------------------------------------------------------
# metric_aliases_for_business_type
# ---------------------------------------------------------------------------


def test_saas_aliases_include_mrr_for_recurring_revenue() -> None:
    aliases = metric_aliases_for_business_type("saas")
    assert "mrr" in aliases["recurring_revenue"]
    assert "recurring_revenue" in aliases["recurring_revenue"]


def test_saas_aliases_case_insensitive_lookup() -> None:
    aliases = metric_aliases_for_business_type("SaaS")
    assert "mrr" in aliases.get("recurring_revenue", ())


# ---------------------------------------------------------------------------
# CanonicalValidationResult diagnostics
# ---------------------------------------------------------------------------


def test_validation_result_has_diagnostics_field() -> None:
    result = CanonicalValidationResult(
        is_valid=True,
        missing_metrics=[],
        diagnostics={"entity_name": "test", "db_metric_names": ["mrr"]},
    )
    assert result.diagnostics["entity_name"] == "test"
    assert "mrr" in result.diagnostics["db_metric_names"]


def test_validation_result_diagnostics_defaults_to_empty_dict() -> None:
    result = CanonicalValidationResult(is_valid=True, missing_metrics=[])
    assert result.diagnostics == {}
