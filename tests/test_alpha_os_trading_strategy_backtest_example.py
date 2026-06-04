from __future__ import annotations

import pytest

from conftest import load_example_module


def test_trading_strategy_backtest_example_rolls_policy_through_prices():
    example = load_example_module("examples/trading_strategy_backtest.py")

    steps = example.backtest_strategy(
        example.EqualWeightLongOnlyStrategy(),
        [
            example.MarketObservation(prices={"BTC": 100.0, "ETH": 50.0}),
            example.MarketObservation(prices={"BTC": 110.0, "ETH": 55.0}),
            example.MarketObservation(prices={"BTC": 99.0, "ETH": 60.5}),
        ],
    )

    assert len(steps) == 2
    assert steps[0].reward == pytest.approx(0.1)
    assert steps[0].equity == pytest.approx(1.1)
    assert steps[0].action == example.PortfolioAction(
        target_weights={"BTC": 0.5, "ETH": 0.5}
    )
    assert steps[0].observation == example.MarketObservation(
        prices={"BTC": 100.0, "ETH": 50.0}
    )
    assert steps[1].reward == pytest.approx(0.0)
    assert steps[1].equity == pytest.approx(1.1)
    assert steps[1].action == example.PortfolioAction(
        target_weights={"BTC": 0.5, "ETH": 0.5}
    )
    assert steps[1].observation == example.MarketObservation(
        prices={"BTC": 110.0, "ETH": 55.0}
    )
