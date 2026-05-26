# Portfolio transition policy boundary

## Problem

Resolved: the persisted `rebalance_friction_policy` object was removed.

The current object does not only describe rebalance friction. It controls how a
desired portfolio target is converted into an executed portfolio target.

The old flow was:

```text
turnover_budget
  -> DecisionBacktestInput
  -> portfolio_trade_transition.TradeTransitionRequest
  -> apply_trade_transition()
```

This is closer to a portfolio or trade transition policy than to a simple
friction assumption.

## Current Field Shape

The remaining transition control is a backtest rollout input:

- `turnover_budget`

The old `friction` object no longer exists in strategy documents, and these
controls are no longer strategy portfolio fields.

## Field Classification

Initial classification:

| Field | Likely responsibility |
|---|---|
| `turnover_budget` | policy / transition constraint |

`turnover_cost_rate` is represented on `TradingEnvironment`, not on the
transition controls.

## RL Analogy

The strategy or policy produces a desired action.

The transition layer decides what action actually happens after applying
turnover budgets.

The world or environment then applies realized costs and rewards.

The remaining fields directly change the action before costs are charged, so
they belong to rollout transition handling rather than the strategy portfolio
shape.

## Risk

The old name encouraged two mistakes:

- treating policy-side action suppression as a simple cost assumption

The lower-level path now uses direct trade-transition inputs rather than a
separate policy object.

## Direction

No further wrapper object is needed for now. Keep the direct fields until a
real transition abstraction becomes necessary.

## Close Condition

Closed when the persisted wrapper was removed and environment cost assumptions
were represented separately.
