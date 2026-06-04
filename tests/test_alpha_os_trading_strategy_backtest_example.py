from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_example() -> ModuleType:
    path = Path("examples/trading_strategy_backtest.py")
    spec = importlib.util.spec_from_file_location("trading_strategy_backtest", path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_trading_strategy_backtest_example_rolls_policy_through_prices():
    example = _load_example()

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
