# Common Strategy Evaluation Boundary

## Problem

alpha-os currently exposes different evaluation paths based on how a candidate
strategy was produced.

Examples:

- hand-written trainless rules use direct strategy evaluation
- discovered signals use the signal discovery evaluation path
- frozen discovery artifacts use fixed-state replay

For an end user, those are provenance differences. They should not change the
meaning of the evaluation result.

## Risk

The evaluation path can become tied to the candidate origin.

That makes it hard to compare:

- a hand-written rule
- a trained model
- an online learner
- an automatically discovered signal set

The user expectation is simpler: once a candidate exists, evaluate it under the
same data, period, cost, baseline, metrics, and report semantics.

## Desired Boundary

Candidate origin should be provenance.

Common evaluation should score a fixed candidate under shared assumptions.

The candidate may come from a manual rule, trained model, online learner, or
signal discovery output, but the final comparison path should have the same
meaning.

## Current Finding

Hand-written trainless candidates and signal-discovery-derived candidates both
feed the decision backtest and report metric machinery.

The remaining gap is not that every metric is totally separate. The gap is that
candidate origin is still partly encoded by execution path names and artifacts:

- hand-written rules route through `candidate_backtest`
- discovered or frozen candidates route through signal discovery execution
- reports expose some artifact refs, but candidate origin is not yet a clear
  report-level fact

## Close Condition

Close this when alpha-os has a clear boundary between:

- candidate production
- common candidate evaluation

and reports make candidate origin visible without changing the meaning of the
evaluation metrics.
