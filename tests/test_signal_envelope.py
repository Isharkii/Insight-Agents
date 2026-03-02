"""
tests/test_signal_envelope.py

Tests for the SignalEnvelope contract proving:
  - Construction and extraction helpers produce correct envelopes
  - Missing optional columns produce deterministic partial insights
  - Explicit warnings are returned for missing optional signals
  - Business-type-aware classification works correctly
  - KPI nodes emit partial envelopes when optional metrics are missing
  - Risk node continues with partial KPI data
  - Pipeline status derives correctly with partial required nodes
  - Backward compatibility with old node_result envelopes
"""

from __future__ import annotations

import unittest
from typing import Any

from agent.signal_envelope import (
    MINIMUM_KPI_SIGNALS,
    MINIMUM_KPI_SIGNALS_BY_TYPE,
    OPTIONAL_KPI_SIGNALS,
    OPTIONAL_KPI_SIGNALS_BY_TYPE,
    EnvelopeStatus,
    classify_kpi_completeness,
    classify_kpi_completeness_for_type,
    envelope_confidence,
    envelope_errors,
    envelope_failed,
    envelope_partial,
    envelope_payload,
    envelope_skipped,
    envelope_status,
    envelope_success,
    envelope_warnings,
    has_minimum_signals,
)
from agent.nodes.node_result import (
    confidence_of,
    errors_of,
    failed,
    partial,
    payload_of,
    skipped,
    status_of,
    success,
    warnings_of,
)


# ===================================================================
# Construction helpers
# ===================================================================


class TestEnvelopeConstruction(unittest.TestCase):
    """envelope_success/partial/skipped/failed produce correct dicts."""

    def test_success_envelope(self) -> None:
        env = envelope_success({"mrr": 12000})
        self.assertEqual(env["status"], "success")
        self.assertEqual(env["payload"]["mrr"], 12000)
        self.assertEqual(env["warnings"], [])
        self.assertEqual(env["errors"], [])
        self.assertEqual(env["confidence_score"], 1.0)

    def test_partial_envelope_with_warnings(self) -> None:
        env = envelope_partial(
            {"mrr": 12000},
            warnings=["ltv not computed"],
            confidence_score=0.7,
        )
        self.assertEqual(env["status"], "partial")
        self.assertIn("ltv not computed", env["warnings"])
        self.assertAlmostEqual(env["confidence_score"], 0.7)

    def test_skipped_envelope(self) -> None:
        env = envelope_skipped("no_data")
        self.assertEqual(env["status"], "skipped")
        self.assertIn("no_data", env["warnings"])
        self.assertEqual(env["confidence_score"], 0.0)

    def test_failed_envelope(self) -> None:
        env = envelope_failed("db_error")
        self.assertEqual(env["status"], "failed")
        self.assertIn("db_error", env["errors"])
        self.assertEqual(env["confidence_score"], 0.0)

    def test_confidence_clamped_to_0_1(self) -> None:
        env = envelope_partial(None, confidence_score=1.5)
        self.assertEqual(env["confidence_score"], 1.0)
        env2 = envelope_partial(None, confidence_score=-0.5)
        self.assertEqual(env2["confidence_score"], 0.0)


# ===================================================================
# Extraction helpers
# ===================================================================


class TestEnvelopeExtraction(unittest.TestCase):
    """Extraction helpers handle both new and old envelope formats."""

    def test_extract_from_full_envelope(self) -> None:
        env = envelope_partial(
            {"mrr": 12000},
            warnings=["ltv missing"],
            errors=["growth_rate failed"],
            confidence_score=0.75,
        )
        self.assertEqual(envelope_status(env), "partial")
        self.assertEqual(envelope_payload(env), {"mrr": 12000})
        self.assertEqual(envelope_warnings(env), ["ltv missing"])
        self.assertEqual(envelope_errors(env), ["growth_rate failed"])
        self.assertAlmostEqual(envelope_confidence(env), 0.75)

    def test_extract_from_old_format(self) -> None:
        """Old node_result envelopes lack warnings/errors/confidence."""
        old = {"status": "success", "payload": {"mrr": 12000}}
        self.assertEqual(envelope_status(old), "success")
        self.assertEqual(envelope_payload(old), {"mrr": 12000})
        self.assertEqual(envelope_warnings(old), [])
        self.assertEqual(envelope_errors(old), [])
        self.assertAlmostEqual(envelope_confidence(old), 0.0)

    def test_extract_from_none(self) -> None:
        self.assertEqual(envelope_status(None), "failed")
        self.assertIsNone(envelope_payload(None))
        self.assertEqual(envelope_warnings(None), [])
        self.assertEqual(envelope_errors(None), [])
        self.assertAlmostEqual(envelope_confidence(None), 0.0)

    def test_extract_from_raw_dict(self) -> None:
        """Dict without status key returns the dict itself as payload."""
        raw = {"mrr": 12000}
        self.assertEqual(envelope_payload(raw), raw)


# ===================================================================
# node_result.py extended helpers
# ===================================================================


class TestNodeResultPartial(unittest.TestCase):
    """node_result compatibility helpers and extraction helpers."""

    def test_partial_status(self) -> None:
        env = partial({"mrr": 12000}, warnings=["ltv missing"])
        self.assertEqual(status_of(env), "success")
        self.assertEqual(payload_of(env), {"mrr": 12000})
        self.assertEqual(warnings_of(env), ["ltv missing"])

    def test_status_of_recognises_partial(self) -> None:
        self.assertEqual(status_of({"status": "partial", "payload": None}), "success")

    def test_backward_compat_success(self) -> None:
        env = success({"mrr": 12000})
        self.assertEqual(status_of(env), "success")
        self.assertEqual(warnings_of(env), [])
        self.assertAlmostEqual(confidence_of(env), 1.0)

    def test_confidence_of(self) -> None:
        env = partial({"mrr": 12000}, confidence_score=0.8)
        self.assertAlmostEqual(confidence_of(env), 0.8)

    def test_errors_of(self) -> None:
        env = partial(None, errors=["missing mrr"])
        self.assertEqual(errors_of(env), ["missing mrr"])


# ===================================================================
# Partial-result evaluation: generic
# ===================================================================


class TestHasMinimumSignals(unittest.TestCase):
    """has_minimum_signals checks for required signal presence."""

    def test_all_minimum_present(self) -> None:
        kpis = {
            "mrr": {"value": 12000, "error": None},
            "churn_rate": {"value": 0.05, "error": None},
        }
        ok, missing = has_minimum_signals(kpis)
        self.assertTrue(ok)
        self.assertEqual(missing, [])

    def test_mrr_missing(self) -> None:
        kpis = {"churn_rate": {"value": 0.05, "error": None}}
        ok, missing = has_minimum_signals(kpis)
        self.assertFalse(ok)
        self.assertIn("mrr", missing)

    def test_mrr_has_error(self) -> None:
        kpis = {
            "mrr": {"value": None, "error": "division_by_zero"},
            "churn_rate": {"value": 0.05, "error": None},
        }
        ok, missing = has_minimum_signals(kpis)
        self.assertFalse(ok)
        self.assertIn("mrr", missing)

    def test_raw_value_accepted(self) -> None:
        kpis = {"mrr": 12000, "churn_rate": 0.05}
        ok, missing = has_minimum_signals(kpis)
        self.assertTrue(ok)


class TestClassifyKPICompleteness(unittest.TestCase):
    """classify_kpi_completeness returns correct (status, warnings, errors, confidence)."""

    def test_all_present_success(self) -> None:
        kpis = {
            "mrr": {"value": 12000, "error": None},
            "churn_rate": {"value": 0.05, "error": None},
            "ltv": {"value": 240000, "error": None},
            "growth_rate": {"value": 0.12, "error": None},
            "arpu": {"value": 100, "error": None},
        }
        status, warnings, errors, confidence = classify_kpi_completeness(kpis)
        self.assertEqual(status, "success")
        self.assertGreaterEqual(confidence, 0.9)

    def test_optional_missing_produces_partial(self) -> None:
        """Missing optional columns produce deterministic partial status."""
        kpis = {
            "mrr": {"value": 12000, "error": None},
            "churn_rate": {"value": 0.05, "error": None},
            "ltv": {"value": None, "error": "insufficient_data"},
            "growth_rate": {"value": 0.12, "error": None},
            "arpu": {"value": 100, "error": None},
        }
        status, warnings, errors, confidence = classify_kpi_completeness(kpis)
        self.assertEqual(status, "partial")
        self.assertGreater(len(warnings), 0)
        self.assertLess(confidence, 1.0)
        self.assertGreater(confidence, 0.5)

    def test_explicit_warnings_for_optional_failures(self) -> None:
        """Explicit warnings are returned for each optional signal failure."""
        kpis = {
            "mrr": {"value": 12000, "error": None},
            "churn_rate": {"value": 0.05, "error": None},
            "ltv": {"value": None, "error": "insufficient_data"},
            "arpu": {"value": None, "error": "division_by_zero"},
        }
        status, warnings, errors, confidence = classify_kpi_completeness(kpis)
        self.assertEqual(status, "partial")
        self.assertTrue(any("ltv" in w for w in warnings))
        self.assertTrue(any("arpu" in w for w in warnings))

    def test_minimum_missing_fails(self) -> None:
        kpis = {"ltv": {"value": 240000, "error": None}}
        status, warnings, errors, confidence = classify_kpi_completeness(kpis)
        self.assertEqual(status, "failed")
        self.assertAlmostEqual(confidence, 0.0)
        self.assertTrue(len(errors) > 0)

    def test_no_optional_signals_is_success(self) -> None:
        """If only minimum signals exist and no optional are present at all."""
        kpis = {
            "mrr": {"value": 12000, "error": None},
            "churn_rate": {"value": 0.05, "error": None},
        }
        status, warnings, errors, confidence = classify_kpi_completeness(kpis)
        self.assertEqual(status, "success")
        self.assertAlmostEqual(confidence, 1.0)


# ===================================================================
# Business-type-aware classification
# ===================================================================


class TestClassifyKPIByBusinessType(unittest.TestCase):
    """classify_kpi_completeness_for_type uses per-type registries."""

    def test_saas_minimum_signals(self) -> None:
        kpis = {
            "mrr": {"value": 12000, "error": None},
            "churn_rate": {"value": 0.05, "error": None},
        }
        status, warnings, errors, confidence = classify_kpi_completeness_for_type(
            kpis, "saas"
        )
        self.assertIn(status, ("success", "partial"))
        self.assertNotEqual(status, "failed")

    def test_ecommerce_minimum_signals(self) -> None:
        kpis = {
            "revenue": {"value": 50000, "error": None},
            "conversion_rate": {"value": 0.03, "error": None},
        }
        status, _, _, _ = classify_kpi_completeness_for_type(kpis, "ecommerce")
        self.assertIn(status, ("success", "partial"))

    def test_ecommerce_missing_required_fails(self) -> None:
        kpis = {
            "aov": {"value": 50, "error": None},
        }
        status, _, errors, confidence = classify_kpi_completeness_for_type(
            kpis, "ecommerce"
        )
        self.assertEqual(status, "failed")
        self.assertAlmostEqual(confidence, 0.0)

    def test_agency_minimum_signals(self) -> None:
        kpis = {
            "total_revenue": {"value": 80000, "error": None},
            "client_churn": {"value": 0.1, "error": None},
        }
        status, _, _, _ = classify_kpi_completeness_for_type(kpis, "agency")
        self.assertIn(status, ("success", "partial"))

    def test_agency_partial_with_missing_optional(self) -> None:
        """Agency has minimum but optional utilization_rate errors."""
        kpis = {
            "total_revenue": {"value": 80000, "error": None},
            "client_churn": {"value": 0.1, "error": None},
            "utilization_rate": {"value": None, "error": "insufficient_data"},
            "revenue_per_employee": {"value": 5000, "error": None},
        }
        status, warnings, errors, confidence = classify_kpi_completeness_for_type(
            kpis, "agency"
        )
        self.assertEqual(status, "partial")
        self.assertTrue(any("utilization_rate" in w for w in warnings))

    def test_unknown_business_type_falls_back(self) -> None:
        kpis = {
            "mrr": {"value": 12000, "error": None},
            "churn_rate": {"value": 0.05, "error": None},
        }
        status, _, _, _ = classify_kpi_completeness_for_type(kpis, "unknown_type")
        self.assertEqual(status, "success")


# ===================================================================
# Pipeline status derivation
# ===================================================================


class TestDerrivePipelineStatus(unittest.TestCase):
    """derive_pipeline_status handles partial required nodes correctly."""

    def test_all_success(self) -> None:
        from agent.graph import derive_pipeline_status

        state: dict[str, Any] = {
            "business_type": "saas",
            "saas_kpi_data": success({"records": []}),
            "risk_data": success({"risk_score": 50}),
        }
        self.assertEqual(derive_pipeline_status(state), "success")

    def test_required_partial_helper_maps_to_success(self) -> None:
        from agent.graph import derive_pipeline_status

        state: dict[str, Any] = {
            "business_type": "saas",
            "saas_kpi_data": partial(
                {"records": []},
                warnings=["ltv missing"],
                confidence_score=0.7,
            ),
            "risk_data": success({"risk_score": 50}),
        }
        self.assertEqual(derive_pipeline_status(state), "success")

    def test_required_failed_yields_failed(self) -> None:
        from agent.graph import derive_pipeline_status

        state: dict[str, Any] = {
            "business_type": "saas",
            "saas_kpi_data": failed("db_error"),
            "risk_data": success({"risk_score": 50}),
        }
        self.assertEqual(derive_pipeline_status(state), "failed")

    def test_optional_failed_yields_partial(self) -> None:
        from agent.graph import derive_pipeline_status

        state: dict[str, Any] = {
            "business_type": "saas",
            "saas_kpi_data": success({"records": []}),
            "risk_data": success({"risk_score": 50}),
            "forecast_data": failed("no forecast model"),
        }
        self.assertEqual(derive_pipeline_status(state), "partial")

    def test_unwired_optional_ignored(self) -> None:
        from agent.graph import derive_pipeline_status

        state: dict[str, Any] = {
            "business_type": "saas",
            "saas_kpi_data": success({"records": []}),
            "risk_data": success({"risk_score": 50}),
            "forecast_data": None,
        }
        self.assertEqual(derive_pipeline_status(state), "success")

    def test_both_required_partial_helpers_map_to_success(self) -> None:
        from agent.graph import derive_pipeline_status

        state: dict[str, Any] = {
            "business_type": "saas",
            "saas_kpi_data": partial(
                {"records": []}, warnings=["ltv missing"],
            ),
            "risk_data": partial(
                {"risk_score": 50}, warnings=["partial kpi"],
            ),
        }
        self.assertEqual(derive_pipeline_status(state), "success")


# ===================================================================
# InsightOutput schema with diagnostics
# ===================================================================


class TestInsightOutputDiagnostics(unittest.TestCase):
    """InsightOutput enforces the competitor-structured contract."""

    def test_structured_contract_serialises(self) -> None:
        from llm_synthesis.schema import InsightOutput

        output = InsightOutput(
            competitive_analysis={
                "summary": "Competitor benchmark summary.",
                "market_position": "Peer market position description.",
                "relative_performance": "Growth metric versus competitor benchmark.",
                "key_advantages": ["ARPU strength versus competitor median."],
                "key_vulnerabilities": ["Churn weakness versus competitor benchmark."],
                "confidence": 0.8,
            },
            strategic_recommendations={
                "immediate_actions": ["Address competitor churn gap immediately."],
                "mid_term_moves": ["Close growth gap versus competitor benchmark."],
                "defensive_strategies": ["Defend against competitor strength in retention metric."],
                "offensive_strategies": ["Exploit competitor weakness in ARPU benchmark."],
            },
        )
        import json
        data = json.loads(output.model_dump_json())
        self.assertIn("competitive_analysis", data)
        self.assertIn("strategic_recommendations", data)
        self.assertEqual(data["competitive_analysis"]["confidence"], 0.8)

    def test_failure_returns_structured_payload(self) -> None:
        from llm_synthesis.schema import InsightOutput

        output = InsightOutput.failure("something broke")
        self.assertEqual(output.competitive_analysis.confidence, 0.0)
        self.assertTrue(
            all(
                item.lower().startswith("conditional:")
                for item in output.strategic_recommendations.immediate_actions
            )
        )


# ===================================================================
# Deterministic partial insights with explicit warnings
# ===================================================================


class TestPartialInsightDeterminism(unittest.TestCase):
    """
    Prove that missing optional columns produce deterministic partial
    insights with explicit warnings — the core contract requirement.
    """

    def test_saas_partial_is_deterministic(self) -> None:
        """Same input always produces the same partial classification."""
        kpis = {
            "mrr": {"value": 12000, "error": None},
            "churn_rate": {"value": 0.05, "error": None},
            "ltv": {"value": None, "error": "insufficient_data"},
            "growth_rate": {"value": 0.12, "error": None},
            "arpu": {"value": None, "error": "division_by_zero"},
        }
        results = [
            classify_kpi_completeness_for_type(kpis, "saas")
            for _ in range(10)
        ]
        # All 10 calls produce the same result
        for r in results:
            self.assertEqual(r[0], results[0][0])
            self.assertAlmostEqual(r[3], results[0][3])
            self.assertEqual(sorted(r[1]), sorted(results[0][1]))

    def test_ecommerce_partial_warnings_explicit(self) -> None:
        """Each missing optional produces an explicit named warning."""
        kpis = {
            "revenue": {"value": 50000, "error": None},
            "conversion_rate": {"value": 0.03, "error": None},
            "aov": {"value": None, "error": "insufficient_data"},
            "cac": {"value": None, "error": "missing_marketing_data"},
            "ltv": {"value": 1000, "error": None},
        }
        status, warnings, errors, confidence = classify_kpi_completeness_for_type(
            kpis, "ecommerce"
        )
        self.assertEqual(status, "partial")
        # Each failed optional produces a warning mentioning its name
        self.assertTrue(any("aov" in w for w in warnings))
        self.assertTrue(any("cac" in w for w in warnings))

    def test_confidence_decreases_with_more_missing_optional(self) -> None:
        """Confidence decreases monotonically as more optional signals fail."""
        base = {
            "mrr": {"value": 12000, "error": None},
            "churn_rate": {"value": 0.05, "error": None},
        }
        # 0 of 3 optional missing
        kpis_full = {
            **base,
            "ltv": {"value": 240000, "error": None},
            "growth_rate": {"value": 0.12, "error": None},
            "arpu": {"value": 100, "error": None},
        }
        _, _, _, conf_full = classify_kpi_completeness_for_type(kpis_full, "saas")

        # 1 of 3 optional missing
        kpis_one = {
            **base,
            "ltv": {"value": None, "error": "err"},
            "growth_rate": {"value": 0.12, "error": None},
            "arpu": {"value": 100, "error": None},
        }
        _, _, _, conf_one = classify_kpi_completeness_for_type(kpis_one, "saas")

        # 2 of 3 optional missing
        kpis_two = {
            **base,
            "ltv": {"value": None, "error": "err"},
            "growth_rate": {"value": None, "error": "err"},
            "arpu": {"value": 100, "error": None},
        }
        _, _, _, conf_two = classify_kpi_completeness_for_type(kpis_two, "saas")

        self.assertGreater(conf_full, conf_one)
        self.assertGreater(conf_one, conf_two)


# ===================================================================
# Signal registry correctness
# ===================================================================


class TestSignalRegistries(unittest.TestCase):
    """Per-type registries are internally consistent."""

    def test_minimum_and_optional_are_disjoint(self) -> None:
        for btype in MINIMUM_KPI_SIGNALS_BY_TYPE:
            required = MINIMUM_KPI_SIGNALS_BY_TYPE[btype]
            optional = OPTIONAL_KPI_SIGNALS_BY_TYPE.get(btype, frozenset())
            overlap = required & optional
            self.assertEqual(overlap, frozenset(), f"Overlap in {btype}: {overlap}")

    def test_all_business_types_have_minimum(self) -> None:
        for btype in ("saas", "ecommerce", "agency"):
            self.assertIn(btype, MINIMUM_KPI_SIGNALS_BY_TYPE)
            self.assertGreater(len(MINIMUM_KPI_SIGNALS_BY_TYPE[btype]), 0)


if __name__ == "__main__":
    unittest.main()
