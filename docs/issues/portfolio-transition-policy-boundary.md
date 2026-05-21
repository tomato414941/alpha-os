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

- `turnover_friction`
- `no_trade_band`
- `execution_cost_aversion`
- `execution_mode`
- `turnover_budget`
- `benefit_scale`
- `min_trade_utility`
- `uncertainty_aversion`
- `risk_aversion`
- `partial_fill_enabled`

These fields do not all belong to one semantic layer.

## Field Classification

Initial classification:

| Field | Likely responsibility |
|---|---|
| `turnover_friction` | mixed: environment cost and policy-side expected cost |
| `no_trade_band` | policy / transition rule |
| `execution_cost_aversion` | policy |
| `execution_mode` | policy |
| `turnover_budget` | policy / transition constraint |
| `benefit_scale` | policy |
| `min_trade_utility` | policy |
| `uncertainty_aversion` | policy |
| `risk_aversion` | policy |
| `partial_fill_enabled` | transition execution / simulator |

## RL Analogy

The strategy or policy produces a desired action.

The transition layer decides what action actually happens after applying
turnover budgets, no-trade bands, utility thresholds, and partial-fill behavior.

The world or environment then applies realized costs and rewards.

`rebalance_friction_policy` currently mixes parts of these layers. It should not
be moved wholesale into world or evaluation config, because several fields
directly change the action before costs are charged.

## Risk

Keeping the current name encourages two mistakes:

- treating policy-side action suppression as a simple cost assumption
- treating environment costs as if they were strategy behavior

It also makes `execution_policy` terminology more confusing, because there is
already a separate `portfolio_execution_policy.ExecutionPolicySpec` that better
describes the action transition behavior.

## Direction

Do not delete this object as if it were only an evaluation cost assumption.

Prefer a future split or rename around a clearer concept such as:

- `portfolio_transition_policy`
- `trade_transition_policy`
- `rebalance_transition_policy`

Before renaming, decide how to separate:

- policy-side expected cost and utility controls
- portfolio transition constraints
- simulator or environment execution effects
- realized cost calculation

## Close Condition

Close this when `rebalance_friction_policy` has been renamed or split so that
policy-side transition behavior is no longer described as generic friction, and
environment cost assumptions are represented separately.
