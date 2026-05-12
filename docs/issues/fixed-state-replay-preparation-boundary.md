# Fixed-State Replay Preparation Boundary

## Problem

`fixed_state_replay` is conceptually different from strategy backtest execution.

It decides which trained or discovered state should be replayed. The backtest
should evaluate an already prepared state.

## Boundary

Treat fixed-state replay as state selection and preparation.

Treat strategy backtest as prepared-state evaluation.

Desired shape:

```text
fixed-state replay preparation
  -> strategy checkpoint
  -> strategy backtest
```

Avoid pushing these responsibilities into strategy backtest:

- finding signal discovery runs
- choosing strategy checkpoints
- reading screening results
- reading compressed beliefs
- deciding which train-period artifacts apply to a test range

## Why It Matters

If strategy backtest also resolves training artifacts, it becomes another heavy
runtime workflow instead of a small evaluation boundary.

That would make lightweight strategy evaluation depend on discovery and DB
artifact layout again.

## Non-Goals

- Do not remove fixed-state replay.
- Do not change the current fixed-state OOS workflow immediately.
- Do not merge signal discovery, screening, and backtest into one abstraction.

## Acceptance Criteria

- The strategy backtest boundary can evaluate an already prepared state.
- Fixed-state replay selection remains outside the backtest function.
- The code path makes it clear which layer prepares state and which layer
  evaluates it.
