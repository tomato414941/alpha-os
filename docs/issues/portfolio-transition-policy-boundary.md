# Portfolio transition policy boundary

## Problem

Resolved: the persisted `rebalance_friction_policy` object and direct
transition budget input were removed.

The old object did not only describe rebalance friction. It controlled how a
desired portfolio target was converted into an executed portfolio target.

The old flow was:

```text
transition budget input
  -> removed decision backtest input
  -> portfolio trade transition layer
```

This is closer to a portfolio or trade transition policy than to a simple
friction assumption.

## Current Field Shape

The old `friction` object, direct transition budget input, and standalone
portfolio trade transition module no longer exist.

## Field Classification

Initial classification:

| Field | Likely responsibility |
|---|---|
| direct action suppression | policy / transition constraint |

The old `turnover_cost_rate` field is no longer represented in current code.
If turnover cost becomes necessary again, it should belong to the environment
or rollout path that actually charges realized costs, not to transition
controls.

## RL Analogy

The strategy or policy produces a desired action.

If a transition layer exists, it decides what action actually happens after
applying explicit action suppression rules.

The world or environment then applies realized costs and rewards.

Action suppression belongs to rollout transition handling rather than the
strategy portfolio shape.

## Risk

The old name encouraged two mistakes:

- treating policy-side action suppression as a simple cost assumption

The lower-level path now uses direct trade-transition inputs rather than a
separate policy object.

## Direction

No further wrapper object is needed for now. Add a transition abstraction only
when a real execution or action-suppression rule needs one.

## Close Condition

Closed when the persisted wrapper was removed.
