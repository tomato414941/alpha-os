# Signal Screening Vs Strategy Evaluation Boundary

## Problem

The current name `signal discovery evaluation` makes a discovery-stage scoring
path sound like a final strategy evaluation path.

That is misleading. The discovery path can generate candidates, screen them,
compress them, and evaluate the selected result inside the same workflow. Those
scores are useful for finding candidates, but they are not the same as a
common strategy backtest or OOS comparison.

## Risk

Exploration-stage scores can be compared directly with hand-written strategy
backtests.

That would mix two different meanings:

- signal screening: scoring used to choose candidate signals during research
- strategy evaluation: scoring a fixed candidate under shared data, cost,
  period, and baseline assumptions

The first has selection bias by construction. The second is the comparison
path that should decide whether a discovered candidate remains useful.

## Current Boundary

Use discovery-stage evaluation as signal or alpha screening.

Use direct strategy evaluation, or a later common evaluation path, for fixed
candidate strategy comparisons.

## Close Condition

Close this when the naming, docs, and run result labels make it clear that
discovery-stage scoring is screening, not final strategy evaluation.
