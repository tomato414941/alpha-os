# Strategy as code boundary

## Problem

`strategy_specs` can become a JSON strategy DSL.

That is risky. A strategy can contain real behavior, state loading, model or
policy code, allocation decisions, and environment assumptions. If all of that
is forced into configuration, the project may create broad generic abstractions
before the real strategy shape is known.

## Current Signal

Checked-in runtime manifests now keep `evaluation_cases` thin, so more
responsibility is visible on `strategy_specs`.

Current strategy documents often include:

- `position_rule_id`
- `family_mix`
- `portfolio_construction`
- `rebalance_friction_policy`
- `execution_policy`
- `holding_cost_policy`
- `signal_discovery_id`

Some of these fields may be part of the strategy. Others look more like
simulator, evaluation environment, cost, provenance, or runtime wiring.

## ML / RL Analogy

In ML systems, the model implementation is usually code. Config selects a model,
dataset, checkpoint, optimizer settings, or evaluation settings; it does not
usually reimplement the model as a large JSON object.

In RL systems, the policy and environment boundary also matters. Policy behavior
belongs with the policy implementation. Market impact, fees, funding, borrow
costs, and simulator behavior belong to the environment or evaluation setup
unless the policy explicitly uses them to decide actions.

## Direction

Do not treat JSON `strategy_specs` as the long-term primary expression of a
strategy implementation.

Prefer:

- strategy behavior in Python code
- strategy state in explicit checkpoints
- manifest entries as thin references to strategy code, strategy parameters,
  checkpoints, data inputs, and evaluation specs
- cost and market simulation assumptions outside the strategy unless the
  strategy directly uses them for decisions

## Risk

Keeping too much behavior in JSON encourages:

- generic strategy abstractions before they are needed
- duplicated strategy variants for small experiment differences
- unclear ownership of execution cost and holding cost assumptions
- confusion between policy behavior, evaluation setup, and provenance

## Next Decision

Before adding new fields to `strategy_specs`, decide whether the field belongs
in:

- strategy code
- a small strategy parameter object
- a strategy checkpoint
- evaluation spec / environment config
- data input or runtime connection config
- provenance / diagnostics

## Close Condition

Close this when new strategies can be represented without growing a broad JSON
strategy DSL, and existing `strategy_specs` have a documented path toward
thinner references.
