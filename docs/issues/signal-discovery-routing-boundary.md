# Signal Discovery Routing Boundary

## Problem

`signal_discovery_id` has historically been used as more than provenance. It
has acted as a routing key across portfolio decision input construction and
some strategy resolution paths.

This makes `SignalDiscovery` look like a primary domain axis even though it is
closer to an internal strategy preparation/search step.

## Risk

Evaluation and downstream workflows currently need to know whether a strategy
came from `SignalDiscovery`.

That creates misleading boundaries:

- strategy overrides rebuild strategies from the referenced signal discovery
  spec
- portfolio decision input construction used to recover subject set through
  compressed belief provenance

ML/RL-style boundaries would usually evaluate a strategy/policy or a checkpoint,
not route through the feature-selection/search job that produced it.

## Boundary

Do not treat `signal_discovery_id` as the general way to decide how a strategy
is evaluated.

Use strategy state as the evaluation boundary. Future checkpoint state should be
introduced only after the checkpoint model is defined. Keep `signal_discovery_id`
as provenance unless a local workflow explicitly needs to load the signal search
specification.

## Current Marker

`portfolio_decision_service.py` no longer resolves subject set through
compressed belief `signal_discovery_id`. Callers pass the resolved `SubjectSet`
explicitly.

## Desired Direction

Evaluation task resolution no longer derives strategy variants from
`signal_discovery_id`. Evaluation planning now prefers direct strategy
evaluation when the strategy is directly executable. Portfolio decision input
construction no longer uses compressed belief provenance to recover subject set.

## Close Condition

Close this when `signal_discovery_id` is no longer a cross-cutting execution
routing key, or when each remaining use is explicitly documented as local
provenance lookup rather than evaluation or workflow routing.
