from __future__ import annotations

from app.services.statistics.market_share_simulation import simulate_market_share


def test_market_share_output_contract_and_determinism() -> None:
    kwargs = {
        "total_industry_size": 1000.0,
        "client_revenue_projection": [120.0, 132.0, 145.2, 159.72],
        "competitor_growth_rates": {
            "comp_a": 0.02,
            "comp_b": [0.01, 0.015, 0.02, 0.025],
        },
        "new_entrant_factor": 0.01,
        "industry_expansion_rate": 0.03,
        "scenario": "base",
    }
    first = simulate_market_share(**kwargs)
    second = simulate_market_share(**kwargs)

    assert first == second
    assert set(first.keys()) == {
        "market_share_series",
        "relative_rank_series",
        "share_delta",
        "competitive_shift_index",
    }
    assert len(first["market_share_series"]) == 4
    assert len(first["relative_rank_series"]) == 4


def test_multi_competitor_simulation_tracks_rank_improvement() -> None:
    output = simulate_market_share(
        total_industry_size=1000.0,
        client_revenue_projection=[150.0, 240.0, 330.0, 420.0, 510.0],
        competitor_growth_rates={
            "comp_a": [0.00, 0.00, 0.00, 0.00, 0.00],
            "comp_b": [0.01, 0.01, 0.01, 0.01, 0.01],
            "comp_c": [0.02, 0.02, 0.02, 0.02, 0.02],
        },
        new_entrant_factor=0.0,
        industry_expansion_rate=0.01,
        scenario="base",
    )

    ranks = output["relative_rank_series"]
    assert ranks[0] > ranks[-1]
    assert output["share_delta"] > 0.0
    assert output["competitive_shift_index"] > 0.0


def test_new_entrant_factor_reduces_client_share() -> None:
    without_entrant = simulate_market_share(
        total_industry_size=900.0,
        client_revenue_projection=[130.0, 140.0, 150.0, 160.0],
        competitor_growth_rates={"comp_a": 0.015, "comp_b": 0.02},
        new_entrant_factor=0.0,
        industry_expansion_rate=0.02,
    )
    with_entrant = simulate_market_share(
        total_industry_size=900.0,
        client_revenue_projection=[130.0, 140.0, 150.0, 160.0],
        competitor_growth_rates={"comp_a": 0.015, "comp_b": 0.02},
        new_entrant_factor=0.04,
        industry_expansion_rate=0.02,
    )

    assert with_entrant["market_share_series"][-1] < without_entrant["market_share_series"][-1]
    assert with_entrant["share_delta"] < without_entrant["share_delta"]


def test_scenario_assumptions_change_competitive_outcome() -> None:
    conservative = simulate_market_share(
        total_industry_size=1200.0,
        client_revenue_projection=[160.0, 175.0, 190.0, 210.0],
        competitor_growth_rates={"comp_a": 0.03, "comp_b": 0.025},
        new_entrant_factor=0.02,
        industry_expansion_rate=0.03,
        scenario="conservative",
    )
    aggressive = simulate_market_share(
        total_industry_size=1200.0,
        client_revenue_projection=[160.0, 175.0, 190.0, 210.0],
        competitor_growth_rates={"comp_a": 0.03, "comp_b": 0.025},
        new_entrant_factor=0.02,
        industry_expansion_rate=0.03,
        scenario="aggressive",
    )

    assert aggressive["share_delta"] > conservative["share_delta"]
    assert aggressive["competitive_shift_index"] > conservative["competitive_shift_index"]

