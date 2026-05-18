# Signal Discovery Routing Boundary

## Problem

`signal_discovery_id` is no longer just provenance. It is used as a routing key
across evaluation, checkpoint preparation, and portfolio decision input
construction.

This makes `SignalDiscovery` look like a primary domain axis even though it is
closer to an internal strategy preparation/search step.

## Risk

Evaluation and downstream workflows currently need to know whether a strategy
came from `SignalDiscovery`.

That creates misleading boundaries:

- evaluation chooses direct vs checkpoint paths from
  `trading_strategy.signal_discovery_id`
- checkpoint preparation groups evaluation tasks by `signal_discovery_id`
- strategy overrides rebuild strategies from the referenced signal discovery
  spec
- portfolio decision input construction recovers subject set through compressed
  belief provenance

ML/RL-style boundaries would usually evaluate a strategy/policy or a checkpoint,
not route through the feature-selection/search job that produced it.

## Boundary

Do not treat `signal_discovery_id` as the general way to decide how a strategy
is evaluated.

Use strategy state and checkpoint state as the evaluation boundary. Keep
`signal_discovery_id` as provenance unless a local workflow explicitly needs to
load the signal search specification.

## Current Suspects

- `evaluation_plan.py` branches on `trading_strategy.signal_discovery_id is None`
- `evaluation_application.py` groups checkpoint preparation by
  `signal_discovery_id`
- `evaluation_task_resolution.py` derives strategy variants from
  `signal_discovery_id`
- `portfolio_decision_service.py` resolves subject set through compressed belief
  `signal_discovery_id`

## Desired Direction

Start with evaluation:

- direct vs checkpoint evaluation should be based on available or required
  strategy checkpoint state, not on whether the strategy has discovery
  provenance
- checkpoint preparation should move toward strategy/preparation inputs rather
  than grouping by `signal_discovery_id`

After that, revisit portfolio decision input construction.

## Close Condition

Close this when `signal_discovery_id` is no longer a cross-cutting execution
routing key, or when each remaining use is explicitly documented as local
provenance lookup rather than evaluation or workflow routing.
