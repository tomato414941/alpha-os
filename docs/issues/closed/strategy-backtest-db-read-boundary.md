# Strategy Backtest DB Read Boundary

Status: Closed

Closed by: `893635f refactor: rename strategy backtest path`

## Issue

`run_strategy_backtest_from_store()` mixes strategy backtest logic with DB reads.

The function currently reads `TradingStrategySpec` and `SubjectSet` through `store`:

- `store.get_trading_strategy(strategy_id)`
- `store.get_subject_set(subject_set_id)`

This makes strategy backtests depend on prior DB registration even when the caller already has the strategy and subject set objects.

## Why This Matters

The DB should not be required just to evaluate an in-memory strategy.

For lightweight hypothesis work, the natural input is:

- `TradingStrategySpec`
- `SubjectSet`
- evaluation date ranges
- evaluation assumptions

Requiring `strategy_id -> DB -> strategy` and `subject_set_id -> DB -> subject set` keeps the heavy manifest/apply/DB path on the critical path.

## Current Boundary

Current path:

```text
strategy_id / subject_set_id
  -> EvaluationStore
  -> TradingStrategySpec / SubjectSet
  -> strategy backtest
```

Desired first step:

```text
TradingStrategySpec / SubjectSet
  -> strategy backtest
```

The old DB-backed function should not remain on the critical path.

## Non-Goals

- Do not split data loading yet.
- Do not introduce a new strategy framework.

## Acceptance Criteria

- A spec-based strategy backtest function exists.
- No DB-backed backtest wrapper remains.
- Existing tests continue to pass.
- No new DB dependency is introduced into the spec-based function.

## Closure Notes

`run_strategy_backtest()` now accepts explicit strategy behavior fields and
`SubjectSet` directly. It no longer accepts `TradingStrategySpec`.

`run_strategy_backtest_from_store()` was later removed with the old DB-backed
evaluation runner path.
