from __future__ import annotations

import app.services.statistics.scenario_orchestration as orchestration_module
from app.services.statistics.scenario_orchestration import (
    orchestrate_all_scenarios,
    orchestrate_scenarios_batch,
    orchestrate_scenario,
    run_combined_scenario,
)


def _base_config() -> dict:
    return {
        "base_case": {
            "revenue_projection": [100.0, 112.0, 125.0, 139.0, 154.0],
        },
        "market_share": {
            "total_industry_size": 1200.0,
            "competitor_growth_rates": {
                "comp_a": 0.02,
                "comp_b": 0.025,
            },
            "new_entrant_factor": 0.01,
            "industry_expansion_rate": 0.02,
            "scenario": "base",
        },
        "erosion": {
            "current_market_share": 0.20,
            "competitive_score": 30.0,
            "industry_growth_rate": 0.08,
            "erosion_factor": 0.12,
            "horizon_periods": 5,
            "severity": "severe",
            "decay_mode": "linear",
        },
        "recession": {
            "gdp_contraction_rate": 0.05,
            "interest_rate_spike": 0.02,
            "industry_sensitivity_coefficient": 1.8,
            "shock_duration_quarters": 2,
            "recovery_curve": "l_shape",
        },
    }


def test_base_scenario_contract_and_determinism() -> None:
    config = _base_config()
    first = orchestrate_scenario(scenario_type="base", config=config)
    second = orchestrate_scenario(scenario_type="base", config=config)

    assert first == second
    assert set(first.keys()) == {
        "scenario_name",
        "revenue_projection",
        "market_share_projection",
        "risk_score",
        "downside_risk_pct",
        "upside_potential_pct",
    }
    assert first["scenario_name"] == "base"
    assert first["revenue_projection"] == [100.0, 112.0, 125.0, 139.0, 154.0]
    assert len(first["market_share_projection"]) == len(first["revenue_projection"])


def test_erosion_scenario_increases_downside_vs_base() -> None:
    config = _base_config()
    baseline = orchestrate_scenario(scenario_type="base", config=config)
    erosion = orchestrate_scenario(scenario_type="erosion", config=config)

    assert sum(erosion["revenue_projection"]) < sum(baseline["revenue_projection"])
    assert erosion["downside_risk_pct"] > baseline["downside_risk_pct"]
    assert erosion["risk_score"] > baseline["risk_score"]


def test_recession_scenario_increases_downside_vs_base() -> None:
    config = _base_config()
    baseline = orchestrate_scenario(scenario_type="base", config=config)
    recession = orchestrate_scenario(scenario_type="recession", config=config)

    assert sum(recession["revenue_projection"]) < sum(baseline["revenue_projection"])
    assert recession["downside_risk_pct"] > baseline["downside_risk_pct"]
    assert recession["risk_score"] > baseline["risk_score"]


def test_combined_scenario_wrapper_and_modular_composition() -> None:
    config = _base_config()
    combined = orchestrate_scenario(scenario_type="combined", config=config)
    wrapped = run_combined_scenario(config)
    erosion = orchestrate_scenario(scenario_type="erosion", config=config)
    recession = orchestrate_scenario(scenario_type="recession", config=config)

    assert combined == wrapped
    assert combined["scenario_name"] == "combined"
    assert len(combined["revenue_projection"]) == 5
    assert len(combined["market_share_projection"]) == 5
    assert combined["downside_risk_pct"] >= erosion["downside_risk_pct"]
    assert combined["downside_risk_pct"] >= recession["downside_risk_pct"]


def test_toggle_override_disables_optional_modules() -> None:
    config = _base_config()
    override = {
        **config,
        "toggles": {
            "use_erosion": False,
            "use_recession": False,
            "use_market_share": False,
        },
    }

    result = orchestrate_scenario(scenario_type="combined", config=override)
    assert result["revenue_projection"] == [100.0, 112.0, 125.0, 139.0, 154.0]
    assert result["market_share_projection"] == []
    assert result["risk_score"] == 0.0
    assert result["downside_risk_pct"] == 0.0
    assert result["upside_potential_pct"] == 0.0


def test_orchestrate_all_scenarios_reuses_single_base_projection(monkeypatch) -> None:
    calls = {"count": 0}

    def _fake_project_base_case(*args, **kwargs):
        calls["count"] += 1
        return {
            "projected_revenue": [90.0, 95.0, 100.0, 105.0],
        }

    monkeypatch.setattr(orchestration_module, "project_base_case", _fake_project_base_case)

    config = _base_config()
    config["base_case"] = {
        "historical_revenue": [80.0, 85.0, 90.0],
        "projection_config": {"method": "cagr", "horizon_quarters": 4},
    }

    outputs = orchestrate_all_scenarios(config=config)
    assert set(outputs.keys()) == {"base", "erosion", "recession", "combined"}
    assert calls["count"] == 1


def test_batch_orchestration_supports_multi_client_payloads() -> None:
    requests = [
        {
            "client_id": "acme",
            "scenario_type": "combined",
            "config": _base_config(),
        },
        {
            "client_id": "beta",
            "scenario_types": ["base", "recession"],
            "config": _base_config(),
        },
    ]

    output = orchestrate_scenarios_batch(requests)
    assert len(output) == 2

    first = output[0]
    assert first["client_id"] == "acme"
    assert first["scenario"]["scenario_name"] == "combined"

    second = output[1]
    assert second["client_id"] == "beta"
    assert set(second["scenarios"].keys()) == {"base", "recession"}
