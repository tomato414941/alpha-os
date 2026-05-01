# Strategy Comparison Contract

This file is the source of truth for the minimum facts required to compare
strategy evaluation results.

## Required Contract

Strategy results are comparable only when they share:

- period
- costs
- required metrics

Required metrics:

- net return
- max drawdown
- Sharpe ratio
- turnover

Extra metrics are allowed.

## Optional Check

Subject set equality is optional.

Use the same-subject-set check when the comparison intends to hold the tradable
universe fixed. Do not require subject set equality when universe selection is
part of the strategy being evaluated.

## Not Part Of This Contract

Baseline is not part of the common comparison contract.

Baseline belongs to promotion or replacement decisions, where the question is
whether a candidate should replace a specific existing strategy.
