from __future__ import annotations

from conftest import load_example_module


def test_trading_strategy_can_return_execution_intent():
    example = load_example_module("examples/trading_strategy_execution_intent.py")

    intent = example.decide_trading_intent(
        example.RiskOffStrategy(),
        example.RiskObservation(
            prices={"BTC": 100.0, "ETH": 50.0},
            current_weights={"BTC": 0.5, "ETH": 0.5},
            risk_score=0.9,
        ),
    )

    assert intent == example.TradingIntent(
        target_weights={"BTC": 0.0, "ETH": 0.0},
        execution=example.ExecutionPreference(urgency="high", order_style="market"),
    )
