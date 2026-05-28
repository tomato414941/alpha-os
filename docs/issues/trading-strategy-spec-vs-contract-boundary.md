# Trading strategy spec vs contract boundary

## Problem

`TradingStrategySpec` is a structured persisted configuration record. It is not
the trading strategy itself.

The glossary defines a trading strategy as a black-box decision component that
consumes observations and optional internal state, then produces trading actions
and optional next strategy state.

Keeping shared fields such as `portfolio_construction` on a single spec can
accidentally require every strategy implementation to share the same internal
shape.

## Direction

Treat `TradingStrategy` as an input/output contract, not as a common data
schema. The strategy should be a black box from the evaluator, backtest runner,
and market/world simulation perspective. Those callers may know the strategy
input and output contract, but they should not interpret internal fields such
as portfolio construction settings as the strategy implementation.

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

This is still not the desired architecture. The current direct strategy
backtest path adapts persisted fields into a backtest recipe instead of running
a `TradingStrategy` black-box contract. It is better described as a temporary
strategy-spec interpreter backtest than as a true trading-strategy backtest.

`evaluation_execution_strategy.py` was removed with the old DB-backed evaluation
runner path.

`strategy_variant.py` was removed after its remaining helper became unused.
This removes the old path that treated signal discovery or compressed belief
provenance as a source for strategy configuration.

`TradingStrategySpec.selection_kind` was removed because a separate selection
mode was redundant.

`TradingStrategySpec.family_mix` was removed. The only active use was a
dual-momentum lookback encoded as a string, which now belongs to the
dual-momentum signal builder as `lookback_steps`.

The `dual_momentum_hold` position rule and its signal builder were removed.
This leaves fewer string-selected strategy behaviors in the temporary backtest
interpreter.

The `constant_hold` position rule was removed. Baseline runs now omit
`position_rule_id` instead of pretending that a no-signal baseline is a strategy
kind.

`TradingStrategySpec.position_rule_id` and `run_strategy_backtest(position_rule_id=...)`
were removed. Backtests now receive an optional precomputed
`position_signal_series_by_subject` instead of selecting strategy behavior from
a string field.

`TradingStrategySpec.rebalance_interval_steps`, `TradingStrategySpec.top_k`,
and the unused `build_trading_strategy_id()` helper were removed. Those values
belong to explicit backtest/allocation inputs, not to the shared strategy
record.

## Close Condition

Close this when strategy execution paths depend on the `TradingStrategy`
contract instead of requiring `TradingStrategySpec` for strategy behavior.
