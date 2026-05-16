# Model vs Policy Evaluation Boundary

## Problem

Alpha OS currently mixes two different evaluation concerns in the same strategy
evaluation path.

In ML/RL terms, these are different jobs:

- model or signal evaluation: whether predictions, beliefs, or signals explain
  the target
- policy or actor evaluation: what happens when the strategy turns those signals
  into portfolio decisions and runs through the environment

Today, `build_signal_discovery_strategy_evaluation_metric_group_results()`
builds datasets, computes prediction diagnostics, runs decision backtests, and
returns both signal-level and policy-level metric groups.

## Risk

When these concerns share one execution path, signal-only metrics can implicitly
require decision backtest inputs. This also makes actor-owned rules such as
candidate selection or `top_k` appear to belong to the evaluator.

That boundary leak makes it harder to tell whether a field belongs to:

- the model or signal state
- the strategy or policy
- the environment/backtest runner
- the evaluator/reporting layer

## Current Finding

Metric groups such as `signed_belief_quality`, `prediction_diagnostics`, and
`portfolio_target_return_alignment` are closer to model/signal evaluation.

Metric groups such as `decision_quality`, `portfolio_concentration`,
`execution_trace`, `cost_drag`, and `robustness` are closer to policy/actor
rollout evaluation.

`EvaluationMetricConfig` currently classifies all supported metric groups as
decision evaluation metric groups, so the implementation does not yet express
this distinction.

## Desired Direction

Separate the conceptual boundary before changing names or extracting new
objects.

The intended shape is:

- model/signal evaluation can run without policy rollout when only signal-level
  metrics are requested
- policy/actor evaluation owns rollout/backtest metrics
- strategy-owned action selection rules do not have to be copied into evaluation
  context just so the evaluator can reconstruct decisions

## Non-Goals

- Do not introduce a new abstraction just to rename the existing mixed path.
- Do not remove policy rollout metrics.
- Do not split every metric group mechanically before the execution boundary is
  clear.

## Close Condition

Close this when model/signal metrics and policy/actor rollout metrics have
explicitly separate ownership, or when the current shared path is documented as
intentional with clear rules for which fields belong to each side.
