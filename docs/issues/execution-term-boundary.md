# Execution Term Boundary

## Problem

`execution` is currently used across several different concepts:

- order or trade execution: converting decisions into orders and fills
- portfolio transition: moving current portfolio state toward desired targets
  under turnover, friction, and cost controls
- strategy run: running a strategy through an engine
- removed `execution_kind` field: previous `trainless` or `trained` behavior
  that was not order execution
- evaluation execution range: the period over which an evaluation engine runs a
  strategy before measuring evaluation ranges

The bare term `execution` can therefore be read as market execution, strategy
run, or evaluation engine execution depending on context.

## Why It Matters

If new code or docs use unqualified `execution`, fields can drift into the
wrong layer.

Examples:

- order-style and fill behavior belong to trade/order execution
- no-trade bands, turnover budgets, and transition utility may belong to
  portfolio transition or rebalance friction
- `trainless` / `trained` were implementation values and
  should not be promoted into a target glossary term
- `backtest_oos` is a current evaluation-spec rigor level, not a run mode
- `execution_range` is an evaluation/run contract field, not an order-execution
  field

## Boundary

Avoid bare `execution` when a more specific term is available.

Prefer scoped terms:

- `order execution` or `trade execution` for orders and fills
- `portfolio transition` or `rebalance transition` for movement from current to
  desired portfolio state
- `strategy run` for running a strategy through an engine
- explicit evaluation job shapes for strict OOS and future checkpoint-based
  evaluation, if checkpoint support returns

Do not introduce `execution kind` or `strategy execution kind` as target terms.
The old `execution_kind` field was removed rather than renamed as a domain
concept. See
[`execution-kind-removal.md`](./execution-kind-removal.md).

Do not introduce `run policy` or `strategy run mode` as target terms. The
old `run_mode` field was removed rather than renamed as a domain concept. See
[`run-mode-removal.md`](./run-mode-removal.md).

## Non-Goals

- Do not rename existing `execution_range` fields immediately.
- Do not split `ExecutionPolicySpec` immediately.
- Do not change manifest compatibility as part of terminology cleanup.

## Acceptance Criteria

- Glossary and design docs avoid bare `execution` where the intended layer is
  market execution, portfolio transition, or strategy run.
- Existing code paths that use `execution` are classified by semantic layer
  before any rename.
- Future fields can tell whether they belong to order execution, portfolio
  transition, strategy requirements, or evaluation/run contracts.
