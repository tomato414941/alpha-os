from __future__ import annotations

from dataclasses import dataclass


def test_trading_strategy_contract_accepts_black_box_decision_component():
    from alpha_os.trading_strategy import TradingStrategy

    @dataclass(frozen=True)
    class ExampleStrategyInput:
        signal: float

    @dataclass(frozen=True)
    class ExampleStrategyOutput:
        target_weight: float

    class FixedWeightStrategy:
        def decide(self, strategy_input: ExampleStrategyInput) -> ExampleStrategyOutput:
            return ExampleStrategyOutput(target_weight=strategy_input.signal)

    strategy: TradingStrategy[ExampleStrategyInput, ExampleStrategyOutput] = (
        FixedWeightStrategy()
    )
    decision = strategy.decide(ExampleStrategyInput(signal=0.5))

    assert decision == ExampleStrategyOutput(target_weight=0.5)
