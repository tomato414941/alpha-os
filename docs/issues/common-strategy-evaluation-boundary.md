# Common Strategy Evaluation Boundary

## Problem

alpha-os needs candidate strategies to be comparable under the same evaluation
contract.

The important question is not where a candidate came from. The important
question is whether it was evaluated with the same comparison conditions:

- data
- period
- subject set
- target
- costs
- portfolio construction
- baseline
- metrics
- report semantics

## Risk

If those comparison conditions are not stable, alpha-os can compare numbers that
look similar but mean different things.

That makes it hard to decide whether one candidate is actually better than
another.

## Boundary

This issue is about the comparison contract for candidate evaluation.

It is separate from:

- `strategy-evaluation-path-boundary.md`, which is about which engine path runs
  a candidate
- `strategy-evaluation-result-explainability.md`, which is about how clearly a
  report explains what was evaluated

## Current Finding

The `crypto_regime_momentum` workflow already checks many comparison-contract
facts through DB artifacts, including subject set, target, cost assumptions,
portfolio construction, metric presence, and decision traces.

The remaining gap is that this is not yet a shared contract across all candidate
types.

## Close Condition

Close this when alpha-os has a small shared comparison contract that can be
checked for candidates regardless of how they were produced.
