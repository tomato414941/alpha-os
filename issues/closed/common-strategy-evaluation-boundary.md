# Common Strategy Evaluation Boundary

Resolved into the active source of truth:
[`docs/design/strategy-comparison-contract.md`](../../design/strategy-comparison-contract.md).

## Problem

alpha-os needs candidate strategies to be comparable under the same evaluation
contract.

The important question is not where a candidate came from. The important
question is whether it was evaluated with the same minimum comparison contract:

- period
- costs
- required metrics
  - net return
  - max drawdown
  - Sharpe ratio
  - turnover

Subject set equality is optional. In many comparisons the tradable universe
should be fixed, but in some strategy comparisons the subject set itself is part
of the strategy being evaluated.

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
facts through DB artifacts, including cost assumptions, metric presence,
decision traces, and subject set equality for that fixed-universe comparison.

The remaining gap is that this is not yet a shared contract across all candidate
types.

## Close Condition

Close this when alpha-os has a small shared check that verifies:

- same evaluation period
- same cost assumptions
- required comparison metrics exist on both compared results

The check may also verify the same subject set when the comparison intends to
hold the tradable universe fixed.
