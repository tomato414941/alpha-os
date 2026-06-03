# Strategy vs Evaluation Universe Boundary

## Problem

The project does not clearly separate universe concepts owned by a strategy from
universe concepts owned by an evaluation.

A strategy can support a broad set of instruments while a specific evaluation
uses only a subset.

Example:

```text
Strategy capability:
  supports ETFs and crypto

Evaluation universe:
  evaluate ETFs only
```

The previous `SubjectSet`-shaped input has been removed. If universe fields
return, the boundary between strategy capability and evaluation condition must
stay explicit.

## Boundary

Trading strategy implementations should describe what the strategy can operate
on and what inputs it needs.

Evaluation inputs should describe what is evaluated in a specific run.

These concepts should be distinguishable:

- strategy-supported universe: what the strategy can handle
- observed/reference instrument set: instruments the strategy may observe but
  not trade
- tradable universe: what the strategy may hold
- evaluation universe: what this evaluation run measures

## Why It Matters

The same strategy should be testable under different evaluation universes
without redefining the strategy.

The strategy may also observe instruments it does not trade.

If alpha-os treats all of these as the same object, strategy definitions can
absorb evaluation policy and evaluation definitions can accidentally become part
of the strategy.

## Non-Goals

- Do not reintroduce `SubjectSet` as the default universe schema.
- Do not introduce a new universe hierarchy until the current usage is mapped.
- Do not promote `input universe` as a term unless the project later proves
  that strategy inputs should be modeled as an instrument-set concept.

## Acceptance Criteria

- The project has glossary entries for the universe terms that should become
  source-of-truth names.
- Trading strategy implementations and evaluation inputs have a documented
  ownership boundary for universe-related fields.
- A future schema change can tell whether a universe field belongs to strategy
  capability, observed/reference instruments, tradable scope, or evaluation
  conditions.
