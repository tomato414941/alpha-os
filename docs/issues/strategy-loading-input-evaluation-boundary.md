# Strategy Loading Input Evaluation Boundary

## Problem

Strategy evaluation currently mixes three responsibilities:

- loading or preparing the executable strategy state
- constructing evaluation inputs or market/world state
- running the strategy rollout and computing metrics

This makes the evaluator aware of details such as checkpoints, selected signals,
screening artifacts, compressed beliefs, and `signal_discovery_id` routing.

## ML/RL Analogy

In supervised ML, the usual split is:

- dataset or dataloader builds model inputs and labels
- model or checkpoint produces predictions
- evaluation computes metrics from predictions and labels

In RL, the usual split is:

- environment/world provides observations, transitions, rewards, and costs
- policy produces actions from observations
- rollout/evaluation runs the policy in the environment and measures the
  trajectory

Trading strategy evaluation is closer to the RL shape:

- market/world/input construction provides observations, features, portfolio
  state, and realized costs
- strategy/policy produces portfolio decisions
- rollout evaluation measures returns, risk, turnover, and decision quality

## Risk

When these responsibilities are mixed, intermediate preparation details become
global domain axes.

Current symptoms include:

- evaluation plans branch on `trading_strategy.signal_discovery_id`
- checkpoint lookup and signal-search provenance leak into evaluation routing
- direct strategies and checkpoint-backed strategies appear to be different
  evaluation types
- `SignalDiscovery` looks more central than the strategy/policy being evaluated

## Boundary

The evaluator should not decide whether a strategy needs a checkpoint or know
how selected signal state was prepared.

Prefer this direction:

- strategy loading/preparation returns an executable strategy or strategy state
- input/environment construction provides the observations and market/world
  state for the requested range
- rollout evaluation runs the executable strategy against those inputs and
  computes metrics

## Current Suspects

- `build_evaluation_plan()` mixes evaluation scheduling with checkpoint lookup
- `evaluation_execution_strategy.py` mixes input construction, strategy state
  loading, rollout execution, and report assembly
- `signal_discovery_id` is used as a routing key instead of local provenance

## Desired Direction

Do not introduce a large new architecture in one step.

Start by identifying existing code blocks that already correspond to:

- strategy loading
- input/environment construction
- rollout evaluation

Then move decisions toward the narrowest owner without adding compatibility
aliases or broad framework abstractions.

## Close Condition

Close this when evaluation code no longer needs to know how a strategy was
prepared, or when the remaining coupling is explicitly documented as local and
intentional.
