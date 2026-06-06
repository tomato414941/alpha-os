from __future__ import annotations

import pytest

from conftest import load_example_module


def test_trading_strategy_can_compose_internal_parts():
    example = load_example_module("examples/trading_strategy_composed.py")

    target = example.decide_portfolio_target(
        example.MomentumAllocatedStrategy(
            alpha_model=example.MomentumAlphaModel(),
            allocator=example.LongOnlyScoreAllocator(),
        ),
        example.MarketObservation(
            features_by_symbol={
                "BTC": {"return_7d": 0.04},
                "ETH": {"return_7d": 0.02},
                "SOL": {"return_7d": -0.01},
            },
            current_weights={},
            equity=1.0,
        ),
    )

    assert target.target_weights["BTC"] == pytest.approx(2.0 / 3.0)
    assert target.target_weights["ETH"] == pytest.approx(1.0 / 3.0)
    assert "SOL" not in target.target_weights
