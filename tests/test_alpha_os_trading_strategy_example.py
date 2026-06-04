from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_example() -> ModuleType:
    path = Path("examples/trading_strategy_rollout.py")
    spec = importlib.util.spec_from_file_location("trading_strategy_rollout", path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_trading_strategy_rollout_example_treats_strategy_as_policy():
    example = _load_example()

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
