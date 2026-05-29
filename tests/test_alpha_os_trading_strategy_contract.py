from __future__ import annotations


def test_trading_strategy_contract_accepts_black_box_decision_component():
    from alpha_os.trading_strategy import (
        TradingStrategy,
        TradingStrategyInput,
        TradingStrategyOutput,
    )

    class FixedWeightStrategy:
        def decide(self, strategy_input: TradingStrategyInput) -> TradingStrategyOutput:
            return TradingStrategyOutput()

    strategy: TradingStrategy[TradingStrategyInput, TradingStrategyOutput] = (
        FixedWeightStrategy()
    )
    decision = strategy.decide(TradingStrategyInput())

    assert isinstance(decision, TradingStrategyOutput)
