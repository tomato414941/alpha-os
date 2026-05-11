# Execution Kind Removal

## Problem

`execution_kind` is a persisted implementation field with values such as
`trainless`, `trained`, and `frozen`.

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

## Current Places To Audit

- `TradingStrategySpec.execution_kind`
- `StrategyExecutionKind`
- `StrategyExecutionSpec.kind`
- `StrategyEvaluationContext.execution_kind`
- manifest and report payloads that persist `execution_kind`
- validation and evaluation-planning branches keyed by `trainless`, `trained`,
  or `frozen`

## Boundary

Do not replace `execution_kind` with another glossary term by default. The first
question is whether the mode is needed at all.

If the behavior remains necessary, separate the concepts explicitly:

- strategy-side requirements, such as whether train-period state is required
- run/evaluation-side state sourcing, such as training per fold or using a fixed
  initial strategy state

## Non-Goals

- Do not break existing manifests or stored reports without a migration path.
- Do not rename code mechanically before the conceptual split is decided.
- Do not reintroduce `execution kind` or `strategy execution kind` as glossary
  terms.

## Acceptance Criteria

- New docs do not use `execution kind` or `strategy execution kind`.
- `execution_kind` is either removed from public payloads or kept only behind a
  documented compatibility layer.
- Evaluation planning no longer depends on a single ambiguous mode if explicit
  requirements and state sources are available.
- Existing persisted artifacts have a migration or compatibility strategy.
