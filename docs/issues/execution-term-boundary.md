# Execution Term Boundary

## Problem

`execution` is currently used across several different concepts:

- order or trade execution: converting decisions into orders and fills
- portfolio transition: moving current portfolio state toward desired targets
  under turnover, friction, and cost controls
- strategy execution: running a strategy through an engine
- strategy execution kind: `trainless`, `trained`, or `frozen` signal-state
  behavior
- evaluation execution range: the period over which an evaluation engine runs a
  strategy before measuring evaluation ranges

The bare term `execution` can therefore be read as market execution, strategy
execution, or evaluation engine execution depending on context.

## Why It Matters

If new code or docs use unqualified `execution`, fields can drift into the
wrong layer.

Examples:

- order-style and fill behavior belong to trade/order execution
- no-trade bands, turnover budgets, and transition utility may belong to
  portfolio transition or rebalance friction
- `trainless` / `trained` / `frozen` belongs to strategy state semantics
- `backtest_oos` / `fixed_state_replay` belongs to run mode
- `execution_range` is an evaluation/run contract field, not an order-execution
  field

## Boundary

Avoid bare `execution` when a more specific term is available.

Prefer scoped terms:

- `order execution` or `trade execution` for orders and fills
- `portfolio transition` or `rebalance transition` for movement from current to
  desired portfolio state
- `strategy execution` for running a strategy through an engine
- `strategy execution kind` for `trainless`, `trained`, and `frozen`
- `run mode` for `backtest_oos`, `fixed_state_replay`, `paper`, and `live`

## Non-Goals

- Do not rename existing `execution_range` fields immediately.
- Do not rename `execution_kind` immediately.
- Do not split `ExecutionPolicySpec` immediately.
- Do not change manifest compatibility as part of terminology cleanup.

## Acceptance Criteria

- Glossary and design docs avoid bare `execution` where the intended layer is
  market execution, portfolio transition, strategy execution, or run mode.
- Existing code paths that use `execution` are classified by semantic layer
  before any rename.
- Future fields can tell whether they belong to order execution, portfolio
  transition, strategy state semantics, or evaluation/run contracts.
