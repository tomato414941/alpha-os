# Portfolio transition policy boundary

## Problem

`rebalance_friction_policy` is too narrow a name for its current behavior.

The current object does not only describe rebalance friction. It controls how a
desired portfolio target is converted into an executed portfolio target.

In code, the flow is:

```text
rebalance_friction_policy
  -> DecisionBacktestInput
  -> portfolio_execution_policy.ExecutionPolicySpec
  -> apply_execution_policy()
```

This is closer to a portfolio or trade transition policy than to a simple
friction assumption.

## Current Field Shape

Current fields include:

- `no_trade_band`
- `execution_cost_aversion`
- `turnover_budget`

These fields are smaller than before, but the object is still named as
`friction` even though it controls portfolio transition behavior.

## Field Classification

Initial classification:

| Field | Likely responsibility |
|---|---|
| `no_trade_band` | policy / transition rule |
| `execution_cost_aversion` | policy |
| `turnover_budget` | policy / transition constraint |

`turnover_cost_rate` is now represented on `TradingEnvironment`, not on
`rebalance_friction_policy`.

## RL Analogy

The strategy or policy produces a desired action.

The transition layer decides what action actually happens after applying
turnover budgets and no-trade bands.

The world or environment then applies realized costs and rewards.

`rebalance_friction_policy` still names policy-side transition behavior as
friction. It should not be moved wholesale into world or evaluation config,
because the remaining fields directly change the action before costs are
charged.

## Risk

Keeping the current name encourages two mistakes:

- treating policy-side action suppression as a simple cost assumption

It also makes `execution_policy` terminology more confusing, because there is
already a separate `portfolio_execution_policy.ExecutionPolicySpec` that better
describes the action transition behavior.

## Direction

Do not delete this object as if it were only an evaluation cost assumption.

Prefer a future split or rename around a clearer concept such as:

- `portfolio_transition_policy`
- `trade_transition_policy`
- `rebalance_transition_policy`

Before renaming, decide whether the remaining fields should become direct
arguments to transition functions instead of another policy object.

## Close Condition

Close this when `rebalance_friction_policy` has been renamed or split so that
policy-side transition behavior is no longer described as generic friction, and
environment cost assumptions are represented separately.
