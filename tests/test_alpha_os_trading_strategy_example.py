from __future__ import annotations

from conftest import load_example_module


def test_trading_strategy_rollout_example_treats_strategy_as_policy():
    example = load_example_module("examples/trading_strategy_rollout.py")

    observations = [
        example.MarketObservation(
            prices={"BTC": 100.0, "ETH": 50.0},
            current_weights={"BTC": 0.0, "ETH": 0.0},
        ),
        example.MarketObservation(
            prices={"BTC": 110.0, "ETH": 0.0},
            current_weights={"BTC": 0.5, "ETH": 0.5},
        ),
    ]

    actions = example.rollout_strategy(
        example.EqualWeightLongOnlyStrategy(), observations
    )

    assert actions == [
        example.PortfolioAction(target_weights={"BTC": 0.5, "ETH": 0.5}),
        example.PortfolioAction(target_weights={"BTC": 1.0}),
    ]
