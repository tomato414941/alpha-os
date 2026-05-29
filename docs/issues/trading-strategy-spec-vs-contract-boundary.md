# Trading strategy spec vs contract boundary

## Problem

`TradingStrategySpec` was a structured persisted configuration record. It was
not the trading strategy itself.

The glossary defines a trading strategy as a black-box decision component that
consumes observations and optional internal state, then produces trading actions
and optional next strategy state.

Keeping shared fields such as `portfolio_construction` on a single spec can
accidentally require every strategy implementation to share the same internal
shape. That spec has now been removed.

## Direction

Treat `TradingStrategy` as an input/output contract, not as a common data
schema. The strategy should be a black box from the evaluator, backtest runner,
and market/world simulation perspective. Those callers may know the strategy
input and output contract, but they should not interpret internal fields such
as portfolio construction settings as the strategy implementation.

Do not reintroduce a shared strategy spec as the domain model for all strategy
implementations.

New strategy implementations should satisfy the `TradingStrategy` protocol and
hide their internal structure unless a specific caller needs it.

## Current Marker

`alpha_os.trading_strategy.TradingStrategy` is the intended black-box strategy
contract.

`TradingStrategy` is a generic protocol. It does not provide universal
`TradingStrategyInput` / `TradingStrategyOutput` marker classes. Concrete
strategies should expose the input and output types they actually use.

`TradingStrategySpec` has been removed. The only remaining code artifact in
`alpha_os.trading_strategy` is the black-box strategy protocol plus concrete
strategy adapters that satisfy it.

`run_strategy_backtest()` was removed. The remaining direct backtest path
accepts already resolved market series instead of interpreting `SubjectSet`
bindings and observation sources inside a strategy-named runner.

This is still not the desired architecture. The current direct backtest path
adapts explicit recipe inputs into a `PortfolioSizingTradingStrategy` instead
of receiving an arbitrary `TradingStrategy` black-box contract.

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

`run_strategy_backtest(top_k=...)` and
`build_direct_strategy_evaluation_metric_group_results(top_k=...)` were removed.
`top_k` now belongs to `PortfolioConstructionSpec`.

`build_direct_strategy_evaluation_metric_group_results()` was removed. That
old direct evaluation entrypoint adapted explicit recipe inputs into a
`PortfolioSizingTradingStrategy` instead of receiving a `TradingStrategy`
black box.

## Close Condition

Close this when strategy execution paths depend on the `TradingStrategy`
contract rather than temporary explicit backtest recipe inputs.
