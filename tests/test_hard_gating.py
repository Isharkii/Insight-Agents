"""Tests for hard gating, uncertainty mode, isolation enforcement, and confidence caps."""
from __future__ import annotations

from agent.nodes.node_result import success, failed
from agent.nodes.synthesis_gate import (
    pre_synthesis_audit,
    should_block_synthesis,
    _MIN_DEPTH,
    _MIN_VALID_LAYERS,
    _MIN_KPI_COVERAGE,
)
from agent.signal_integrity import UnifiedSignalIntegrity
from app.services.statistics.signal_conflict import (
    build_uncertainty_summary,
    detect_conflicts,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_kpi_payload(
    metrics: list[str],
    depth: int = 6,
    source: str = "formula",
) -> dict:
    """Build a minimal KPI payload with the given depth per metric."""
    records = []
    for i in range(depth):
        computed = {
            m: {"value": float(100 + i), "source": source}
            for m in metrics
        }
        records.append({"computed_kpis": computed})
    return {
        "metrics": metrics,
        "records": records,
    }


def _make_forecast_payload(
    r2: float = 0.8,
    slope: float = 0.05,
    input_points: int = 12,
) -> dict:
    """Build a minimal forecast payload."""
    return {
        "forecasts": {
            "revenue": {
                "forecast_data": {
                    "status": "ok",
                    "input_points": input_points,
                    "regression": {
                        "r_squared": r2,
                        "slope": slope,
                    },
                    "forecast": {f"month_{i}": float(100 + i) for i in range(3)},
                },
            },
        },
    }


def _healthy_state() -> dict:
    """State with all layers healthy — should pass all gates."""
    return {
        "business_type": "saas",
        "entity_name": "test",
        "pipeline_status": "success",
        "saas_kpi_data": success(
            _make_kpi_payload(["mrr", "churn_rate", "arpu", "ltv"], depth=6)
        ),
        "forecast_data": success(_make_forecast_payload(r2=0.8)),
        "cohort_data": success({
            "cohort_keys": ["signup_month"],
            "cohorts_by_key": {
                "signup_month": {"count": 3},
            },
        }),
        "segmentation": success({
            "top_contributors": [
                {"name": "enterprise"},
                {"name": "smb"},
            ],
        }),
        "risk_data": success({
            "risk_score": 35,
            "risk_level": "moderate",
        }),
        "signal_conflicts": success({
            "conflict_result": {
                "conflicts": [],
                "conflict_count": 0,
                "total_severity": 0.0,
                "confidence_penalty": 0.0,
                "uncertainty_flag": False,
                "status": "clean",
            },
        }),
    }


# ── pre_synthesis_audit tests ────────────────────────────────────────────


class TestPreSynthesisAudit:
    def test_healthy_state_passes(self):
        state = _healthy_state()
        audit = pre_synthesis_audit(state)
        assert audit["status"] == "pass"
        assert audit["missing_requirements"] == []
        assert audit["uncertainty_mode"] is False

    def test_insufficient_depth_degrades(self):
        state = _healthy_state()
        # Replace KPI with depth=2 (< 3 ideal) — soft block, not hard block
        state["saas_kpi_data"] = success(
            _make_kpi_payload(["mrr", "churn_rate", "arpu", "ltv"], depth=2)
        )
        audit = pre_synthesis_audit(state)
        assert audit["status"] == "degraded"
        assert any("depth" in req for req in audit["missing_requirements"])

    def test_zero_depth_blocks(self):
        state = _healthy_state()
        # depth=0 is a hard block — no time-series data at all
        state["saas_kpi_data"] = success(
            _make_kpi_payload(["mrr", "churn_rate", "arpu", "ltv"], depth=0)
        )
        audit = pre_synthesis_audit(state)
        assert audit["status"] == "blocked"

    def test_insufficient_layers_partial_insight(self):
        # Only KPI, no other layers → partial_insight (not blocked)
        state = {
            "business_type": "saas",
            "entity_name": "test",
            "pipeline_status": "success",
            "saas_kpi_data": success(
                _make_kpi_payload(["mrr", "churn_rate"], depth=6)
            ),
            "risk_data": success({"risk_score": 20, "risk_level": "low"}),
        }
        audit = pre_synthesis_audit(state)
        assert audit["status"] == "degraded"
        assert audit["insight_quality"] == "partial_insight"

    def test_zero_valid_layers_blocks(self):
        # No valid layers at all → hard block
        state = {
            "business_type": "saas",
            "entity_name": "test",
            "pipeline_status": "failed",
        }
        audit = pre_synthesis_audit(state)
        assert audit["status"] == "blocked"
        assert audit["insight_quality"] == "blocked"

    def test_incomplete_kpi_coverage_degrades(self):
        state = _healthy_state()
        # Only 1 out of 4 metrics has data
        state["saas_kpi_data"] = success({
            "metrics": ["mrr", "churn_rate", "arpu", "ltv"],
            "records": [{"computed_kpis": {"mrr": {"value": 100, "source": "formula"}}}],
        })
        audit = pre_synthesis_audit(state)
        # Coverage = 1/4 = 0.25 < 0.5 → degraded (soft block, not hard block)
        assert audit["status"] == "degraded"
        assert any("coverage" in req.lower() for req in audit["missing_requirements"])

    def test_forecast_unusable_with_alternatives_degrades(self):
        state = _healthy_state()
        # Make forecast unusable but keep other layers
        state["forecast_data"] = failed("model_failed", {})
        audit = pre_synthesis_audit(state)
        # Has 3+ layers (kpi, cohort, segmentation) → alternatives exist
        # Forecast unusable alone doesn't block if alternatives exist
        # But the confidence cap will reduce score
        assert audit["status"] in ("pass", "degraded")

    def test_forecast_unusable_without_alternatives_degrades_to_partial(self):
        state = {
            "business_type": "saas",
            "entity_name": "test",
            "pipeline_status": "partial",
            "saas_kpi_data": success(
                _make_kpi_payload(["mrr", "churn_rate"], depth=6)
            ),
            "forecast_data": failed("model_failed", {}),
        }
        audit = pre_synthesis_audit(state)
        # 1 valid layer (KPI) + forecast unusable → partial_insight, not blocked
        assert audit["status"] == "degraded"
        assert audit["insight_quality"] == "partial_insight"

    def test_high_conflict_severity_triggers_uncertainty_mode(self):
        state = _healthy_state()
        state["signal_conflicts"] = success({
            "conflict_result": {
                "conflicts": [
                    {
                        "signal_a": "revenue_growth_delta",
                        "signal_b": "churn_delta",
                        "severity": 0.9,
                    },
                    {
                        "signal_a": "revenue_growth_delta",
                        "signal_b": "conversion_delta",
                        "severity": 0.7,
                    },
                ],
                "conflict_count": 2,
                "total_severity": 1.6,
                "confidence_penalty": 0.19,
                "uncertainty_flag": True,
                "status": "conflicts_detected",
            },
        })
        audit = pre_synthesis_audit(state)
        assert audit["uncertainty_mode"] is True
        assert audit["conflict_summary"] is not None
        assert audit["conflict_summary"]["decision"] == "withheld"

    def test_low_conflict_severity_no_uncertainty(self):
        state = _healthy_state()
        state["signal_conflicts"] = success({
            "conflict_result": {
                "conflicts": [
                    {
                        "signal_a": "slope",
                        "signal_b": "deviation_percentage",
                        "severity": 0.5,
                    },
                ],
                "conflict_count": 1,
                "total_severity": 0.5,
                "confidence_penalty": 0.06,
                "uncertainty_flag": False,
                "status": "conflicts_detected",
            },
        })
        audit = pre_synthesis_audit(state)
        assert audit["uncertainty_mode"] is False


# ── should_block_synthesis tests ─────────────────────────────────────────


class TestShouldBlockSynthesis:
    def test_healthy_state_not_blocked(self):
        state = _healthy_state()
        assert should_block_synthesis(state) is False

    def test_pipeline_failed_blocks(self):
        state = _healthy_state()
        state["pipeline_status"] = "failed"
        assert should_block_synthesis(state) is True

    def test_depth_gate_degrades_not_blocks(self):
        state = _healthy_state()
        state["saas_kpi_data"] = success(
            _make_kpi_payload(["mrr", "churn_rate", "arpu", "ltv"], depth=2)
        )
        # depth=2 is a soft block → degraded synthesis allowed
        assert should_block_synthesis(state) is False

    def test_zero_depth_gate_blocks(self):
        state = _healthy_state()
        state["saas_kpi_data"] = success(
            _make_kpi_payload(["mrr", "churn_rate", "arpu", "ltv"], depth=0)
        )
        assert should_block_synthesis(state) is True

    def test_partial_insight_not_blocked_by_conflicts(self):
        """Gate 4 (conflict blocking) should be bypassed for partial_insight.

        A sparse-data state with 2 valid layers produces partial_insight.
        Even with 3+ signal conflicts, synthesis should NOT be blocked —
        the conflicts are surfaced as warnings, not hard gates.
        """
        state = {
            "business_type": "saas",
            "saas_kpi_data": success(
                _make_kpi_payload(["mrr", "churn_rate", "arpu", "ltv"], depth=2)
            ),
            "risk_data": success({"risk_score": 0.5, "risk_level": "medium"}),
            "segmentation": success({
                "top_contributors": [{"name": "a"}, {"name": "b"}],
            }),
            # Inject conflicts with count >= 3 and high severity
            "signal_conflicts": success({
                "conflict_result": {
                    "conflict_count": 4,
                    "total_severity": 2.5,
                    "uncertainty_flag": True,
                    "confidence_penalty": 0.3,
                    "conflicts": [
                        {"signal_a": "a", "signal_b": "b", "severity": 0.9},
                        {"signal_a": "c", "signal_b": "d", "severity": 0.8},
                        {"signal_a": "e", "signal_b": "f", "severity": 0.5},
                        {"signal_a": "g", "signal_b": "h", "severity": 0.3},
                    ],
                    "warnings": [],
                },
            }),
        }
        # Verify it's partial_insight (< 3 valid layers)
        audit = pre_synthesis_audit(state)
        assert audit["insight_quality"] == "partial_insight"
        # Despite 4 conflicts, should NOT block
        assert should_block_synthesis(state) is False


# ── Confidence caps tests ────────────────────────────────────────────────


class TestConfidenceCaps:
    def test_insufficient_layers_caps_at_025(self):
        state = {
            "business_type": "saas",
            "saas_kpi_data": success(
                _make_kpi_payload(["mrr", "churn_rate", "arpu"], depth=12)
            ),
        }
        integrity = UnifiedSignalIntegrity.compute(state)
        # Only KPI layer → 1 scoring layer → capped at 0.25
        assert integrity["valid_layer_count"] < 3
        assert integrity["overall_score"] <= 0.25
        assert integrity["confidence_breakdown"]["final"] <= 0.25
        assert len(integrity["confidence_breakdown"]["penalties"]["caps_applied"]) > 0

    def test_forecast_unusable_caps_at_030(self):
        state = {
            "business_type": "saas",
            "saas_kpi_data": success(
                _make_kpi_payload(["mrr", "churn_rate", "arpu"], depth=12)
            ),
            "forecast_data": failed("model_error", {}),
            "cohort_data": success({
                "cohort_keys": ["month"],
                "cohorts_by_key": {"month": {"count": 5}},
            }),
            "segmentation": success({
                "top_contributors": [{"name": "a"}, {"name": "b"}],
            }),
        }
        integrity = UnifiedSignalIntegrity.compute(state)
        assert integrity["forecast_usable"] is False
        assert integrity["overall_score"] <= 0.30

    def test_healthy_state_no_caps(self):
        state = _healthy_state()
        integrity = UnifiedSignalIntegrity.compute(state)
        caps = integrity["confidence_breakdown"]["penalties"]["caps_applied"]
        # Healthy state should have no caps applied
        assert caps == []

    def test_breakdown_base_gte_final(self):
        """Base score should always be >= final (caps only reduce)."""
        state = {
            "business_type": "saas",
            "saas_kpi_data": success(
                _make_kpi_payload(["mrr"], depth=3)
            ),
        }
        integrity = UnifiedSignalIntegrity.compute(state)
        breakdown = integrity["confidence_breakdown"]
        assert breakdown["base"] >= breakdown["final"]

    def test_convenience_fields_present(self):
        state = _healthy_state()
        integrity = UnifiedSignalIntegrity.compute(state)
        assert "kpi_depth" in integrity
        assert "kpi_coverage_ratio" in integrity
        assert "valid_layer_count" in integrity
        assert "forecast_usable" in integrity
        assert "confidence_breakdown" in integrity


# ── build_uncertainty_summary tests ──────────────────────────────────────


class TestBuildUncertaintySummary:
    def test_produces_structured_output(self):
        conflict_result = {
            "conflicts": [
                {"signal_a": "revenue_growth_delta", "signal_b": "churn_delta", "severity": 0.9},
                {"signal_a": "ltv_delta", "signal_b": "cac_delta", "severity": 0.7},
            ],
            "conflict_count": 2,
            "total_severity": 1.6,
        }
        summary = build_uncertainty_summary(conflict_result)
        assert summary["decision"] == "withheld"
        assert "revenue_growth_delta" in summary["affected_signals"]
        assert "churn_delta" in summary["affected_signals"]
        assert "1.6" in summary["conflict_summary"] or "1.60" in summary["conflict_summary"]

    def test_empty_conflicts(self):
        summary = build_uncertainty_summary({
            "conflicts": [],
            "conflict_count": 0,
            "total_severity": 0.0,
        })
        assert summary["decision"] == "withheld"
        assert summary["affected_signals"] == []


# ── Isolation enforcement tests ──────────────────────────────────────────


class TestIsolationEnforcement:
    def test_isolated_forecast_excluded_from_integrity_scoring(self):
        state = {
            "business_type": "saas",
            "saas_kpi_data": success(
                _make_kpi_payload(["mrr", "churn_rate", "arpu"], depth=12)
            ),
            "forecast_data": success({
                "forecasts": {
                    "revenue": {
                        "forecast_data": {
                            "status": "ok",
                            "input_points": 12,
                            "regression": {
                                "r_squared": 0.05,  # very low → isolated
                                "slope": 0.02,
                            },
                        },
                    },
                },
            }),
        }
        integrity = UnifiedSignalIntegrity.compute(state)
        assert "forecast" in integrity["isolated_layers"]
        assert integrity["forecast_usable"] is False


# ── Insight quality + layer classification tests ─────────────────────────


class TestInsightQuality:
    def test_full_insight_with_3_layers(self):
        state = _healthy_state()
        integrity = UnifiedSignalIntegrity.compute(state)
        assert integrity["insight_quality"] == "full_insight"
        assert integrity["valid_layer_count"] >= 3

    def test_partial_insight_with_1_layer(self):
        state = {
            "business_type": "saas",
            "saas_kpi_data": success(
                _make_kpi_payload(["mrr", "churn_rate", "arpu"], depth=12)
            ),
        }
        integrity = UnifiedSignalIntegrity.compute(state)
        assert integrity["insight_quality"] == "partial_insight"
        assert integrity["valid_layer_count"] < 3

    def test_partial_insight_with_2_layers(self):
        state = {
            "business_type": "saas",
            "saas_kpi_data": success(
                _make_kpi_payload(["mrr", "churn_rate", "arpu"], depth=12)
            ),
            "forecast_data": success(_make_forecast_payload(r2=0.8)),
        }
        integrity = UnifiedSignalIntegrity.compute(state)
        assert integrity["insight_quality"] == "partial_insight"
        assert integrity["valid_layer_count"] == 2

    def test_blocked_with_zero_layers(self):
        state = {"business_type": "saas"}
        integrity = UnifiedSignalIntegrity.compute(state)
        assert integrity["insight_quality"] == "blocked"
        assert integrity["valid_layer_count"] == 0

    def test_layer_classification_keys(self):
        state = _healthy_state()
        integrity = UnifiedSignalIntegrity.compute(state)
        classification = integrity["layer_classification"]
        assert set(classification.keys()) == {"kpi", "forecast", "competitive", "cohort", "segmentation"}
        # In healthy state, most should be valid
        valid_count = sum(1 for v in classification.values() if v == "valid")
        assert valid_count >= 3

    def test_layer_classification_isolated(self):
        state = {
            "business_type": "saas",
            "saas_kpi_data": success(
                _make_kpi_payload(["mrr", "churn_rate", "arpu"], depth=12)
            ),
            "forecast_data": success({
                "forecasts": {
                    "revenue": {
                        "forecast_data": {
                            "status": "ok",
                            "input_points": 12,
                            "regression": {"r_squared": 0.05, "slope": 0.02},
                        },
                    },
                },
            }),
        }
        integrity = UnifiedSignalIntegrity.compute(state)
        assert integrity["layer_classification"]["forecast"] == "isolated"
        assert integrity["layer_classification"]["kpi"] == "valid"

    def test_partial_insight_not_blocked_by_should_block(self):
        """1-2 valid layers should NOT hard-block synthesis."""
        state = {
            "business_type": "saas",
            "entity_name": "test",
            "pipeline_status": "partial",
            "saas_kpi_data": success(
                _make_kpi_payload(["mrr", "churn_rate", "arpu"], depth=6)
            ),
            "forecast_data": success(_make_forecast_payload(r2=0.8)),
            "risk_data": success({"risk_score": 20, "risk_level": "low"}),
        }
        # 2 valid layers (kpi + forecast) → partial_insight, NOT blocked
        assert should_block_synthesis(state) is False

    def test_audit_insight_quality_field(self):
        state = _healthy_state()
        audit = pre_synthesis_audit(state)
        assert audit["insight_quality"] == "full_insight"
        assert "layer_classification" in audit["audit_details"]


# ── Missing signal detection tests ────────────────────────────────────


class TestMissingSignalDetection:
    def test_healthy_state_no_missing_signals(self):
        state = _healthy_state()
        integrity = UnifiedSignalIntegrity.compute(state)
        report = integrity["missing_signal_report"]
        required_missing = [
            s for s in report["missing_signals"]
            if s["classification"] == "required"
        ]
        assert required_missing == []
        assert report["required_missing_count"] == 0

    def test_missing_required_signal_detected(self):
        # State with KPI but no risk_data (required for saas)
        state = {
            "business_type": "saas",
            "saas_kpi_data": success(
                _make_kpi_payload(["mrr", "churn_rate", "arpu"], depth=6)
            ),
        }
        integrity = UnifiedSignalIntegrity.compute(state)
        report = integrity["missing_signal_report"]
        assert report["required_missing_count"] >= 1
        required_names = [
            s["signal"] for s in report["missing_signals"]
            if s["classification"] == "required"
        ]
        assert "risk" in required_names

    def test_missing_optional_signal_detected(self):
        state = _healthy_state()
        # Remove forecast (optional for saas)
        state.pop("forecast_data", None)
        integrity = UnifiedSignalIntegrity.compute(state)
        report = integrity["missing_signal_report"]
        optional_names = [
            s["signal"] for s in report["missing_signals"]
            if s["classification"] == "optional"
        ]
        assert "forecast" in optional_names

    def test_failed_required_signal_detected(self):
        state = _healthy_state()
        state["risk_data"] = failed("crash", {})
        integrity = UnifiedSignalIntegrity.compute(state)
        report = integrity["missing_signal_report"]
        required_names = [
            s["signal"] for s in report["missing_signals"]
            if s["classification"] == "required"
        ]
        assert "risk" in required_names
        # Check status is "failed"
        risk_entry = next(
            s for s in report["missing_signals"]
            if s["signal"] == "risk"
        )
        assert risk_entry["status"] == "failed"

    def test_missing_required_penalises_confidence(self):
        # Compare confidence with and without required signal
        state_full = _healthy_state()
        integrity_full = UnifiedSignalIntegrity.compute(state_full)

        state_missing = _healthy_state()
        state_missing.pop("risk_data", None)  # remove required signal
        # Need to re-add as missing (not present at all)
        integrity_missing = UnifiedSignalIntegrity.compute(state_missing)

        # Missing required signal should have lower confidence
        # (penalty of 0.10 per missing required)
        report = integrity_missing["missing_signal_report"]
        assert report["required_missing_count"] >= 1
        caps = integrity_missing["confidence_breakdown"]["penalties"]["caps_applied"]
        missing_caps = [c for c in caps if c.get("cap") == "missing_required_signals"]
        assert len(missing_caps) > 0

    def test_report_structure(self):
        state = _healthy_state()
        integrity = UnifiedSignalIntegrity.compute(state)
        report = integrity["missing_signal_report"]
        assert "missing_signals" in report
        assert "required_missing_count" in report
        assert "optional_missing_count" in report
        assert "actions_taken" in report
        assert isinstance(report["missing_signals"], list)
        assert isinstance(report["actions_taken"], list)

    def test_missing_signal_does_not_silently_default_to_zero(self):
        """Missing signals should be explicitly reported, not silently zeroed."""
        state = {
            "business_type": "saas",
            "saas_kpi_data": success(
                _make_kpi_payload(["mrr", "churn_rate"], depth=6)
            ),
        }
        integrity = UnifiedSignalIntegrity.compute(state)
        report = integrity["missing_signal_report"]
        # risk_data is required for saas but missing
        assert report["required_missing_count"] >= 1
        # Actions should document what was done
        assert any("missing" in a or "penalised" in a for a in report["actions_taken"])


# ── Spec compliance: blocked output format ────────────────────────────


class TestBlockedOutputFormat:
    def test_blocked_state_has_spec_fields(self):
        """Blocked output must include pipeline_status, block_reasons, eligible_for_llm."""
        from agent.nodes.synthesis_gate import synthesis_gate_node

        state = {
            "business_type": "saas",
            "entity_name": "test",
            "pipeline_status": "failed",
        }
        result = synthesis_gate_node(state)
        assert result["synthesis_blocked"] is True
        assert result["eligible_for_llm"] is False
        assert result["pipeline_status"] == "blocked"
        assert isinstance(result["block_reasons"], list)
        assert len(result["block_reasons"]) > 0

    def test_blocked_reasons_include_pipeline_failed(self):
        from agent.nodes.synthesis_gate import synthesis_gate_node

        state = {
            "business_type": "saas",
            "entity_name": "test",
            "pipeline_status": "failed",
        }
        result = synthesis_gate_node(state)
        assert any("pipeline_status" in r for r in result["block_reasons"])

    def test_blocked_reasons_include_depth_for_zero_depth(self):
        from agent.nodes.synthesis_gate import synthesis_gate_node

        state = _healthy_state()
        state["saas_kpi_data"] = success(
            _make_kpi_payload(["mrr", "churn_rate", "arpu", "ltv"], depth=0)
        )
        result = synthesis_gate_node(state)
        assert result["synthesis_blocked"] is True
        assert any("depth" in r or "data points" in r for r in result["block_reasons"])

    def test_passed_state_has_eligible_for_llm(self):
        from agent.nodes.synthesis_gate import synthesis_gate_node

        state = _healthy_state()
        result = synthesis_gate_node(state)
        assert result["synthesis_blocked"] is False
        assert result["eligible_for_llm"] is True

    def test_all_block_conditions_enforced(self):
        """Each hard-block condition must produce synthesis_blocked=True.

        Soft-block conditions (depth 1-2, coverage > 0% but < 50%) degrade
        but do NOT block — they allow partial synthesis with hedged tone.
        """
        # Condition 1: pipeline_status == failed
        assert should_block_synthesis({
            "business_type": "saas",
            "entity_name": "test",
            "pipeline_status": "failed",
        }) is True

        # Condition 2: KPI depth == 0 (hard block)
        s = _healthy_state()
        s["saas_kpi_data"] = success(
            _make_kpi_payload(["mrr", "churn_rate", "arpu", "ltv"], depth=0)
        )
        assert should_block_synthesis(s) is True

        # Condition 2b: KPI depth 1-2 is a soft block → does NOT block
        s_soft = _healthy_state()
        s_soft["saas_kpi_data"] = success(
            _make_kpi_payload(["mrr", "churn_rate", "arpu", "ltv"], depth=2)
        )
        assert should_block_synthesis(s_soft) is False

        # Condition 3: KPI coverage == 0% (hard block)
        s2 = _healthy_state()
        s2["saas_kpi_data"] = success({
            "metrics": ["mrr", "churn_rate", "arpu", "ltv"],
            "records": [],
        })
        assert should_block_synthesis(s2) is True

        # Condition 3b: KPI coverage > 0% but < 50% is soft → does NOT block
        s2_soft = _healthy_state()
        s2_soft["saas_kpi_data"] = success({
            "metrics": ["mrr", "churn_rate", "arpu", "ltv"],
            "records": [{"computed_kpis": {"mrr": {"value": 100, "source": "formula"}}}],
        })
        assert should_block_synthesis(s2_soft) is False

        # Condition 4: 0 valid layers
        assert should_block_synthesis({
            "business_type": "saas",
            "entity_name": "test",
            "pipeline_status": "success",
        }) is True

        # Condition 5: severity >= 1.0 AND uncertainty_flag
        s3 = _healthy_state()
        s3["signal_conflicts"] = success({
            "conflict_result": {
                "conflicts": [
                    {"signal_a": "a", "signal_b": "b", "severity": 0.9},
                    {"signal_a": "c", "signal_b": "d", "severity": 0.7},
                ],
                "conflict_count": 2,
                "total_severity": 1.6,
                "confidence_penalty": 0.19,
                "uncertainty_flag": True,
                "status": "conflicts_detected",
            },
        })
        assert should_block_synthesis(s3) is True


# ── Spec compliance: centralized layer filter ─────────────────────────


class TestCentralizedLayerFilter:
    def test_healthy_state_all_eligible(self):
        state = _healthy_state()
        result = UnifiedSignalIntegrity.filter_layers_for_downstream(state)
        assert "kpi" in result["eligible_layers"]
        assert len(result["isolated_layers"]) == 0

    def test_isolated_forecast_excluded(self):
        state = {
            "business_type": "saas",
            "saas_kpi_data": success(
                _make_kpi_payload(["mrr", "churn_rate", "arpu"], depth=12)
            ),
            "forecast_data": success({
                "forecasts": {
                    "revenue": {
                        "forecast_data": {
                            "status": "ok",
                            "input_points": 12,
                            "regression": {"r_squared": 0.05, "slope": 0.02},
                        },
                    },
                },
            }),
        }
        result = UnifiedSignalIntegrity.filter_layers_for_downstream(state)
        assert "forecast" not in result["eligible_layers"]
        assert "forecast" in result["isolated_layers"]

    def test_degraded_excluded_when_not_allowed(self):
        state = {
            "business_type": "saas",
            "saas_kpi_data": success(
                _make_kpi_payload(["mrr", "churn_rate", "arpu"], depth=12)
            ),
            "forecast_data": success({
                "forecasts": {
                    "revenue": {
                        "forecast_data": {
                            "status": "ok",
                            "input_points": 12,
                            "regression": {"r_squared": 0.25, "slope": 0.02},
                        },
                    },
                },
            }),
        }
        result_with = UnifiedSignalIntegrity.filter_layers_for_downstream(
            state, allow_degraded=True
        )
        result_without = UnifiedSignalIntegrity.filter_layers_for_downstream(
            state, allow_degraded=False
        )
        # Degraded layers should appear in one but not the other
        assert len(result_with["eligible_layers"]) >= len(result_without["eligible_layers"])


# ── Final Confidence Cap Enforcement ──────────────────────────────────────


class TestFinalConfidenceCaps:
    """Verify enforce_final_confidence_caps acts as a safety net."""

    def test_no_caps_when_healthy(self):
        integrity = {
            "layer_classification": {"kpi": "valid", "forecast": "valid"},
            "valid_layer_count": 3,
            "forecast_usable": True,
        }
        trace = UnifiedSignalIntegrity.enforce_final_confidence_caps(0.85, integrity)
        assert trace["raw_confidence"] == 0.85
        assert trace["final_confidence"] == 0.85
        assert trace["applied_caps"] == []

    def test_kpi_degraded_cap(self):
        integrity = {
            "layer_classification": {"kpi": "degraded"},
            "valid_layer_count": 3,
            "forecast_usable": True,
        }
        trace = UnifiedSignalIntegrity.enforce_final_confidence_caps(0.80, integrity)
        assert trace["final_confidence"] <= 0.35
        assert len(trace["applied_caps"]) == 1
        assert trace["applied_caps"][0]["cap"] == "kpi_degraded"

    def test_forecast_unusable_cap(self):
        integrity = {
            "layer_classification": {"kpi": "valid"},
            "valid_layer_count": 3,
            "forecast_usable": False,
        }
        trace = UnifiedSignalIntegrity.enforce_final_confidence_caps(0.60, integrity)
        assert trace["final_confidence"] <= 0.30
        assert any(c["cap"] == "forecast_unusable" for c in trace["applied_caps"])

    def test_insufficient_layers_cap(self):
        integrity = {
            "layer_classification": {"kpi": "valid"},
            "valid_layer_count": 2,
            "forecast_usable": True,
        }
        trace = UnifiedSignalIntegrity.enforce_final_confidence_caps(0.50, integrity)
        assert trace["final_confidence"] <= 0.25
        assert any(c["cap"] == "insufficient_layers" for c in trace["applied_caps"])

    def test_multiple_caps_most_restrictive_wins(self):
        integrity = {
            "layer_classification": {"kpi": "degraded"},
            "valid_layer_count": 1,
            "forecast_usable": False,
        }
        trace = UnifiedSignalIntegrity.enforce_final_confidence_caps(0.90, integrity)
        # All 3 caps apply; insufficient_layers (0.25) is most restrictive
        assert trace["final_confidence"] <= 0.25
        assert len(trace["applied_caps"]) == 3

    def test_already_below_cap_no_change(self):
        integrity = {
            "layer_classification": {"kpi": "degraded"},
            "valid_layer_count": 3,
            "forecast_usable": True,
        }
        trace = UnifiedSignalIntegrity.enforce_final_confidence_caps(0.20, integrity)
        assert trace["final_confidence"] == 0.20
        assert trace["applied_caps"] == []

    def test_trace_format_matches_spec(self):
        integrity = {
            "layer_classification": {"kpi": "valid"},
            "valid_layer_count": 2,
            "forecast_usable": True,
        }
        trace = UnifiedSignalIntegrity.enforce_final_confidence_caps(0.50, integrity)
        assert "raw_confidence" in trace
        assert "applied_caps" in trace
        assert "final_confidence" in trace
        assert isinstance(trace["applied_caps"], list)


# ── KPI Coverage Validation ───────────────────────────────────────────────


class TestKPICoverageValidation:
    """Verify KPI coverage gating blocks when coverage < 50%."""

    def test_full_coverage_sufficient(self):
        state = {
            "business_type": "saas",
            "saas_kpi_data": success(
                _make_kpi_payload(["mrr", "churn_rate", "arpu", "ltv"], depth=6)
            ),
        }
        report = UnifiedSignalIntegrity.validate_kpi_coverage(state)
        assert report["coverage_ratio"] == 1.0
        assert report["sufficient"] is True
        assert report["impact"] is None
        assert report["missing_metrics"] == []

    def test_below_50_pct_blocks(self):
        # Only 1 of 4 metrics has data
        payload = {
            "metrics": ["mrr", "churn_rate", "arpu", "ltv"],
            "records": [
                {"computed_kpis": {"mrr": {"value": 100.0, "source": "formula"}}},
            ],
        }
        state = {
            "business_type": "saas",
            "saas_kpi_data": success(payload),
        }
        report = UnifiedSignalIntegrity.validate_kpi_coverage(state)
        assert report["coverage_ratio"] < 0.5
        assert report["sufficient"] is False
        assert report["impact"] == "insufficient KPI reliability"
        assert len(report["missing_metrics"]) >= 2

    def test_exactly_50_pct_passes(self):
        # 2 of 4 metrics
        payload = {
            "metrics": ["mrr", "churn_rate", "arpu", "ltv"],
            "records": [
                {"computed_kpis": {
                    "mrr": {"value": 100.0, "source": "formula"},
                    "churn_rate": {"value": 0.05, "source": "formula"},
                }},
            ],
        }
        state = {
            "business_type": "saas",
            "saas_kpi_data": success(payload),
        }
        report = UnifiedSignalIntegrity.validate_kpi_coverage(state)
        assert report["coverage_ratio"] >= 0.5
        assert report["sufficient"] is True

    def test_no_kpi_data_zero_coverage(self):
        state = {"business_type": "saas"}
        report = UnifiedSignalIntegrity.validate_kpi_coverage(state)
        assert report["coverage_ratio"] == 0.0
        assert report["sufficient"] is False

    def test_coverage_report_in_integrity_compute(self):
        state = _healthy_state()
        integrity = UnifiedSignalIntegrity.compute(state)
        assert "kpi_coverage_report" in integrity
        assert integrity["kpi_coverage_report"]["sufficient"] is True

    def test_coverage_degrades_synthesis(self):
        """When KPI coverage > 0% but < 50%, synthesis_gate degrades (not blocks)."""
        payload = {
            "metrics": ["mrr", "churn_rate", "arpu", "ltv"],
            "records": [
                {"computed_kpis": {"mrr": {"value": 100.0, "source": "formula"}}},
            ],
        }
        state = _healthy_state()
        state["saas_kpi_data"] = success(payload)
        audit = pre_synthesis_audit(state)
        assert audit["status"] == "degraded"
        has_coverage_reason = any(
            "KPI coverage" in req or "kpi_coverage" in req.lower() or "coverage" in req.lower()
            for req in audit.get("missing_requirements", [])
        )
        assert has_coverage_reason

    def test_zero_coverage_blocks_synthesis(self):
        """When KPI coverage == 0%, synthesis_gate should hard-block."""
        payload = {
            "metrics": ["mrr", "churn_rate", "arpu", "ltv"],
            "records": [],
        }
        state = _healthy_state()
        state["saas_kpi_data"] = success(payload)
        audit = pre_synthesis_audit(state)
        assert audit["status"] == "blocked"


# ── Benchmark-Based Recommendations ──────────────────────────────────────


class TestBenchmarkRecommendations:
    """Verify benchmark intelligence feeds into prioritization."""

    def test_benchmark_snapshot_missing(self):
        from agent.helpers.signal_snapshots import benchmark_signal_snapshot
        state = {}
        snap = benchmark_signal_snapshot(state)
        assert snap["status"] == "missing"
        assert snap["weakest_metrics"] == []
        assert snap["strongest_metrics"] == []

    def test_benchmark_snapshot_extracts_position(self):
        from agent.helpers.signal_snapshots import benchmark_signal_snapshot
        state = {
            "benchmark_data": success({
                "composite": {
                    "overall_score": 70.0,
                    "growth_score": 60.0,
                    "stability_score": 55.0,
                },
                "market_position": {
                    "position": "Leader",
                    "confidence": 0.85,
                },
                "ranking": {
                    "metric_rankings": {
                        "mrr": {"percentile_rank": 80.0},
                        "churn_rate": {"percentile_rank": 25.0},
                        "arpu": {"percentile_rank": 90.0},
                    },
                },
                "peer_selection": {"selected_peers": ["a", "b", "c"]},
            }),
        }
        snap = benchmark_signal_snapshot(state)
        assert snap["status"] == "success"
        assert snap["market_position"] == "Leader"
        assert snap["peer_count"] == 3
        assert len(snap["weakest_metrics"]) >= 1
        assert snap["weakest_metrics"][0]["metric"] == "churn_rate"

    def test_prioritization_includes_benchmark_recommendations(self):
        from agent.nodes.prioritization_node import prioritization_node
        state = _healthy_state()
        state["root_cause"] = None
        state["benchmark_data"] = success({
            "composite": {
                "overall_score": 70.0,
                "growth_score": 60.0,
                "stability_score": 55.0,
            },
            "market_position": {
                "position": "Leader",
                "confidence": 0.85,
            },
            "ranking": {
                "metric_rankings": {
                    "mrr": {"percentile_rank": 80.0},
                    "churn_rate": {"percentile_rank": 25.0},
                },
            },
            "peer_selection": {"selected_peers": ["a", "b"]},
        })
        result = prioritization_node(state)
        prio = result["prioritization"]
        assert prio["benchmark_used"] is True
        assert prio["benchmark_market_position"] == "Leader"
        assert len(prio["benchmark_recommendations"]) >= 1

    def test_benchmark_recommendations_withheld_in_uncertainty(self):
        from agent.nodes.prioritization_node import prioritization_node
        state = _healthy_state()
        state["root_cause"] = None
        state["benchmark_data"] = success({
            "composite": {"overall_score": 70.0, "growth_score": 60.0, "stability_score": 55.0},
            "market_position": {"position": "Leader", "confidence": 0.85},
            "ranking": {"metric_rankings": {}},
            "peer_selection": {"selected_peers": ["a", "b"]},
        })
        # Set high conflict severity to trigger uncertainty_mode
        state["signal_conflicts"] = success({
            "conflict_result": {
                "conflicts": [
                    {"type": "a", "severity": 0.6},
                    {"type": "b", "severity": 0.6},
                ],
                "conflict_count": 2,
                "total_severity": 1.2,
                "confidence_penalty": 0.15,
                "uncertainty_flag": True,
                "status": "detected",
                "warnings": [],
            },
        })
        result = prioritization_node(state)
        prio = result["prioritization"]
        assert prio["uncertainty_mode"] is True
        assert prio["benchmark_recommendations"] == []

    def test_benchmark_recommendations_not_withheld_for_moderate_conflicts(self):
        from agent.nodes.prioritization_node import prioritization_node

        state = _healthy_state()
        state["root_cause"] = None
        state["benchmark_data"] = success({
            "composite": {
                "overall_score": 70.0,
                "growth_score": 60.0,
                "stability_score": 55.0,
            },
            "market_position": {
                "position": "Leader",
                "confidence": 0.85,
            },
            "ranking": {
                "metric_rankings": {
                    "mrr": {"percentile_rank": 20.0},
                    "churn_rate": {"percentile_rank": 75.0},
                },
            },
            "peer_selection": {"selected_peers": ["a", "b"]},
        })
        # Moderate conflicts: severity can be >1.0 in aggregate, but without
        # high-severity uncertainty flag and with <3 conflicts, recommendations
        # should still be produced (hedged by confidence), not fully withheld.
        state["signal_conflicts"] = success({
            "conflict_result": {
                "conflicts": [
                    {"type": "a", "severity": 0.6},
                    {"type": "b", "severity": 0.6},
                ],
                "conflict_count": 2,
                "total_severity": 1.2,
                "confidence_penalty": 0.15,
                "uncertainty_flag": False,
                "status": "detected",
                "warnings": [],
            },
        })

        result = prioritization_node(state)
        prio = result["prioritization"]
        assert prio["uncertainty_mode"] is False
        assert prio["decision"] == "active"
        assert len(prio["benchmark_recommendations"]) >= 1
