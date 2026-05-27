# Backtest SubjectSet Input Boundary

## Problem

`run_strategy_backtest()` currently accepts `SubjectSet` directly.

That makes the backtest runner responsible for interpreting subject bindings,
observation specs, assets, instruments, and feature-plane construction before it
can evaluate strategy behavior.

This is too broad for a backtest execution function.

## Why It Matters

`SubjectSet` describes what can be observed or bound to subjects. It is not the
same thing as the market/world input used by one backtest run.

If `run_strategy_backtest()` consumes `SubjectSet` directly, it mixes:

- input/environment construction
- universe or subject binding interpretation
- strategy behavior selection
- rollout/evaluation

This makes it harder to move toward a Trading Strategy / policy boundary where
the evaluator runs a strategy against already prepared market inputs.

## ML/RL Analogy

In ML, evaluation usually receives a dataset or dataloader output, not a schema
object that it must interpret into samples.

In RL, rollout evaluation runs a policy in an environment. The evaluator should
not also be responsible for resolving the environment definition from a registry
or binding spec.

For alpha-os, the analogous split should be:

- input/environment construction resolves `SubjectSet` into market data
- strategy/policy produces portfolio decisions
- backtest/evaluation measures the resulting trajectory

## Current Marker

`run_strategy_backtest()` accepts:

```python
subject_set: SubjectSet
```

and calls `build_subject_set_feature_planes()` internally.

## Desired Direction

Move `SubjectSet` interpretation out of `run_strategy_backtest()`.

Prefer passing already prepared market/world inputs into the backtest runner.
The exact object does not need to be introduced immediately, but the boundary
should make clear that `SubjectSet` belongs to input construction, not rollout
evaluation.

## Non-Goals

- Do not introduce a large environment framework just to fix this.
- Do not rename `SubjectSet` as a substitute for moving the responsibility.
- Do not add another spec object unless the existing call sites prove it is
  needed.

## Acceptance Criteria

- `run_strategy_backtest()` no longer accepts `SubjectSet` directly.
- Subject binding and observation loading happen before the backtest runner.
- The backtest runner receives market/world input that is already resolved for
  the requested evaluation.
