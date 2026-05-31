# Policy Rollout Evaluation Boundary

## Problem

Alpha OS does not yet have a clear RL-style policy evaluation boundary.

A trading strategy is increasingly treated as a policy-like component: it
receives observations and state, then produces trading actions or portfolio
decisions. Evaluating that component should mean evaluating what happens when it
interacts with the market or backtest environment.

This policy rollout concern used to be mixed with predictor diagnostics and
signal-level metrics.

## Why This Matters

Policy rollout evaluation should answer:

- what actions or portfolio decisions did the strategy produce?
- what return, risk, turnover, cost, drawdown, or concentration resulted?
- how did market and execution assumptions affect the outcome?

It should not need to own predictor-quality metrics unless the evaluation is
explicitly asking for both predictor quality and rollout behavior.

## Current Finding

`decision_quality` is currently the only default metric group left for policy
or actor rollout evaluation.

`prediction_diagnostics` has been removed from the ordinary strategy evaluation
metric groups. Predictor-quality metrics still need an explicit owner and
execution path.

## Desired Direction

Make policy rollout evaluation a distinct responsibility from predictor
evaluation.

The backtest path should evaluate strategy behavior in an environment. Predictor
validation should be upstream or separately requested.

## Close Condition

Close this when strategy or policy rollout metrics have an explicit owner and
execution path, or when the project deliberately documents why they remain mixed
with predictor evaluation.
