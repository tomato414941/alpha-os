# Cost Assumption Boundary

## Problem

Cost can mean different things in a trading system:

1. realized execution cost
2. evaluation cost assumption
3. strategy estimated cost

These are not the same responsibility.

Realized execution cost is what actually happened after trading.

Evaluation cost assumption is the cost model used to calculate backtest or OOS
net returns.

Strategy estimated cost is an input to the strategy's decision process, such as
avoiding trades when expected spread, slippage, funding, or borrow cost is too
high.

In RL terms, realized cost belongs to the world or environment: it is part of
the reward calculation after an actor takes an action.

Expected or predicted cost belongs to the policy or actor side when it changes
the decision before the action is taken. It is not the same object as the
realized cost charged by the environment.

## Risk

If these meanings are mixed, alpha-os can blur:

- strategy logic
- evaluation assumptions
- execution records

That makes it hard to tell whether a cost value is part of the alpha hypothesis,
part of the evaluation contract, or part of realized trading state.

## Boundary

Use these meanings:

- realized execution cost: execution or broker records
- evaluation cost assumption: backtest or OOS evaluation contract
- strategy estimated cost: strategy rule input

Decision rule:

- realized cost used to calculate net results belongs to evaluation
- estimated cost used to decide whether or how to trade belongs to strategy

## Current Suspects

- `ExecutionCostAssumptionsSpec` carries fields such as `fee_bps`,
  `market_impact_bps`, and spread assumptions. These look like world or
  simulator assumptions, but they can also be used by policy-side trade utility.
- `HoldingCostAssumptionsSpec` carries funding and borrow assumptions. These are
  closer to realized environment costs.

The code should not rely on a generic cost object to stand for both expected
decision cost and realized environment cost.

## Close Condition

Close this when alpha-os can distinguish these cost meanings in code or
documents without relying on the generic word `cost`.
