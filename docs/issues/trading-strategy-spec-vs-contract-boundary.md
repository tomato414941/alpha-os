# Trading strategy spec vs contract boundary

## Problem

`TradingStrategySpec` is a structured persisted configuration record. It is not
the trading strategy itself.

The glossary defines a trading strategy as a black-box decision component that
consumes observations and optional internal state, then produces trading actions
and optional next strategy state.

Keeping shared fields such as `position_rule_id`, `family_mix`,
`portfolio_construction`, `selection_kind`, and `rebalance_interval_steps` on a
single spec can accidentally require every strategy implementation to share the
same internal shape.

## Direction

Treat `TradingStrategy` as an input/output contract, not as a common data
schema.

`TradingStrategySpec` should remain only where a persisted manifest-style record
is still needed. It should not become the domain model for all strategy
implementations.

New strategy implementations should satisfy the `TradingStrategy` protocol and
hide their internal structure unless a specific caller needs it.

## Current Marker

`alpha_os.trading_strategy.TradingStrategy` is the intended black-box strategy
contract.

`TradingStrategyInput` and `TradingStrategyOutput` are intentionally empty for
now. They mark the contract boundary without making trading strategy depend on
the pre-existing `PortfolioDecisionInput` / `PortfolioDecisionOutput` types.

`run_strategy_backtest()` no longer accepts `TradingStrategySpec`; it accepts
the explicit behavior fields it needs.

`evaluation_execution_strategy.py` no longer imports `TradingStrategySpec` or
uses helper functions typed around it. It still reads the persisted strategy
record and adapts the fields needed by the current backtest path.

`strategy_variant.py` was removed after its remaining helper became unused.
This removes the old path that treated signal discovery or compressed belief
provenance as a source for strategy configuration.

## Close Condition

Close this when strategy execution paths depend on the `TradingStrategy`
contract instead of requiring `TradingStrategySpec` for strategy behavior.
