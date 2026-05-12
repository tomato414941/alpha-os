# Execution Kind Removal

## Problem

`execution_kind` was a persisted implementation field with values such as
`trainless` and `trained`.

The name should not become a domain term:

- `execution` already means converting trading decisions into orders and fills
- `execution_kind` is not about order or trade execution
- `execution kind` and `strategy execution kind` have been removed from the
  glossary
- the current field mixes strategy-side requirements with run/evaluation-side
  state sourcing

## Goal

Remove `execution_kind` as a public/domain-facing concept.

Before replacing it, decide whether the underlying classification is needed at
all. If the behavior is still needed, represent the actual requirements
directly instead of preserving a single mode-like field.

## Completed Removal

- `TradingStrategySpec.execution_kind` removed
- `StrategyExecutionKind` removed
- `StrategyExecutionSpec` removed
- `StrategyEvaluationContext.execution_kind` removed from the request context;
  executor input-source routing remains tracked separately
- manifest and report payloads no longer persist `execution_kind`
- evaluation planning no longer branches on `trainless` or `trained`

## Boundary

Do not replace `execution_kind` with another glossary term by default. The first
question is whether the mode is needed at all.

If the behavior remains necessary, separate the concepts explicitly:

- strategy-side requirements, such as whether train-period state is required
- run/evaluation-side state sourcing, such as training per fold or using a fixed
  strategy checkpoint

## Acceptance Criteria

- New docs do not use `execution kind` or `strategy execution kind`.
- `execution_kind` is removed from public payloads.
- Evaluation planning no longer depends on a single ambiguous mode if explicit
  requirements and state sources are available.
