# Strategy environment cost boundary

## Problem

`TradingStrategySpec.portfolio` currently carries `execution_policy` and
`holding_cost_policy`.

Those names make the fields look strategy-owned, but their current main use is
to feed evaluation cost assumptions:

```text
TradingStrategySpec.portfolio.execution_policy
  -> ExecutionCostAssumptionsSpec
  -> backtest net-return / cost-drag calculation

TradingStrategySpec.portfolio.holding_cost_policy
  -> HoldingCostAssumptionsSpec
  -> backtest funding / borrow cost calculation
```

This mixes strategy or policy state with world / environment assumptions.

## RL Analogy

In an RL-shaped system:

- the agent or policy decides actions
- the world or environment applies fills, fees, slippage, funding, borrow costs,
  and other consequences
- the evaluator measures rewards and diagnostics from the interaction

Market impact, fees, spread, funding, and borrow costs are environment-side
unless the strategy explicitly observes or estimates them before choosing an
action.

If a strategy is cost-aware, the cost estimate should be modeled as an input to
the policy. That is still different from the realized or simulated cost charged
by the environment after the action.

## Current Signal

`execution_policy` in `TradingStrategySpec` contains:

- `market_impact_bps`
- `fee_bps`
- `bid_ask_spread_bps`

`holding_cost_policy` contains:

- `funding_bps_per_step`
- `borrow_fee_bps_per_step`

These are currently closer to `ExecutionCostAssumptionsSpec` and
`HoldingCostAssumptionsSpec` than to strategy behavior.

## Risk

Keeping these fields on the strategy can make an evaluated strategy look like it
includes the world it was evaluated in.

That makes it harder to tell whether a result changed because the policy changed
or because the environment cost assumptions changed.

## Direction

Do not add more environment cost fields to `TradingStrategySpec`.

Before moving fields, classify each use as one of:

- policy input used before choosing an action
- portfolio transition rule used to modify an action
- environment cost charged after an action
- evaluation metric or diagnostic assumption

Long term, prefer environment or evaluation-owned cost assumptions for backtest
net-return calculation. Keep strategy-owned cost fields only when the strategy
actually uses the value to decide actions.

## Close Condition

Close this when `execution_policy` and `holding_cost_policy` are either moved
out of `TradingStrategySpec.portfolio` or explicitly documented as policy inputs
with environment-side cost assumptions represented separately.
