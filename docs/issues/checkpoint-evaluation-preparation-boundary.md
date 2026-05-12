# Checkpoint Evaluation Preparation Boundary

## Problem

Checkpoint-based evaluation is conceptually different from strategy backtest
execution.

It decides which trained or discovered state should be replayed. The backtest
should evaluate an already prepared state.

The current walk-forward evaluation use case also owns preparation orchestration:

- resolve evaluation tasks
- decide whether a strategy requires prepared state
- check whether fold checkpoints already exist
- run the signal discovery workflow when checkpoints are missing
- persist `SignalDiscoveryRun`
- persist `StrategyCheckpoint`
- evaluate the prepared checkpoints

In ML/RL terms, this mixes training or preparation with evaluation. Evaluation
should consume a prepared checkpoint; a separate preparation step should produce
that checkpoint.

## Boundary

Treat checkpoint evaluation as state selection and preparation.

Treat strategy backtest as prepared-state evaluation.

Treat checkpoint creation as a preparation concern, not as an evaluation
concern.

Desired shape:

```text
checkpoint evaluation preparation
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

If the evaluation use case owns missing-checkpoint preparation, `requires_signal_train`
also becomes ambiguous: it can mean either "this strategy needs a checkpoint for
evaluation" or "this evaluation run should train/prepare missing checkpoints."

## Non-Goals

- Do not remove checkpoint-based evaluation.
- Do not change the current fixed-state OOS workflow immediately.
- Do not merge signal discovery, screening, and backtest into one abstraction.
- Do not replace `execution_kind` with another broad mode flag.

## Acceptance Criteria

- The strategy backtest boundary can evaluate an already prepared state.
- Checkpoint selection remains outside the backtest function.
- Checkpoint creation is separated from evaluation execution.
- The code path makes it clear which layer prepares state and which layer
  evaluates it.
