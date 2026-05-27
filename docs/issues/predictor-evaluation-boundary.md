# Predictor Evaluation Boundary

## Problem

Alpha OS does not yet have a clear evaluation boundary for predictors.

Some existing metrics, such as `prediction_diagnostics`, check whether signal or
model outputs explain a prediction target. That is closer to ML model
validation than to trading strategy rollout evaluation.

Today those metrics are built inside the strategy evaluation path, so predictor
quality is coupled to decision backtest execution.

## Why This Matters

Predictor evaluation should answer:

- does the predictor output relate to the prediction target?
- is the sign, ranking, or bucket spread useful?
- is the output sufficiently covered across subjects and dates?

It should not require portfolio construction, execution costs, or rollout
state unless those are explicitly part of the predictor being evaluated.

## Current Finding

`prediction_diagnostics` currently computes signal-vs-forward-return metrics,
including correlation, hit rate, long/short bucket spread, and coverage.

Those metrics are useful, but they are currently grouped with strategy
evaluation metrics.

## Desired Direction

Separate predictor evaluation from strategy or policy rollout evaluation.

Predictor evaluation should be able to run independently when the question is
only whether a signal, model, or predictor output explains the prediction
target.

## Close Condition

Close this when predictor metrics have an explicit owner and execution path, or
when the project deliberately documents why predictor evaluation remains inside
strategy evaluation.
