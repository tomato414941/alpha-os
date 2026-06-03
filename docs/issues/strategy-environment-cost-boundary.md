# Strategy environment cost boundary

## Problem

`TradingStrategySpec` used to carry `TradingEnvironment`.

Those names make the fields look strategy-owned, but their current main use is
to feed evaluation cost assumptions:

```text
TradingStrategySpec.trading_environment
  -> backtest net-return / cost-drag calculation

TradingStrategySpec.trading_environment
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

The previous `TradingEnvironment` value object has been removed from the
current code because it was not connected to an actual backtest, rollout, or
market simulation path.

The underlying boundary is still valid: market impact, fees, spread, funding,
and borrow costs are environment-side trading costs, not strategy behavior.

## Risk

Keeping these fields on the strategy can make an evaluated strategy look like it
includes the world it was evaluated in.

That makes it harder to tell whether a result changed because the policy changed
or because the environment cost assumptions changed.

## Direction

Do not add more environment cost fields to `TradingStrategySpec`.

`TradingStrategySpec.trading_environment` has been removed. Do not reintroduce
a standalone cost DTO just to preserve this concept.

Before moving fields, classify each use as one of:

- policy input used before choosing an action
- portfolio transition rule used to modify an action
- environment cost charged after an action
- evaluation metric or diagnostic assumption

Long term, introduce an environment-owned representation only together with the
evaluation or market interaction code that actually applies fills, costs, and
rewards. Keep strategy-owned cost fields only when the strategy actually uses
the value to decide actions.

## Close Condition

Close this when remaining strategy/environment cost coupling is either removed
or explicitly documented as intentional.
