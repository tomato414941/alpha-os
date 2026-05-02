# Strategy Portfolio Responsibility Boundary

## Problem

In trading, a portfolio usually means the holdings or allocation at a point in
time.

`TradingStrategySpec.portfolio` does not represent current holdings. It
also does more than clean portfolio allocation. The current
`StrategyPortfolioSpec` carries a mixed set of strategy and evaluation-facing
settings:

- portfolio construction
- selection
- execution assumptions
- rebalance friction assumptions
- holding cost assumptions

The issue is not only the field name. A simple rename would still leave one
container holding allocation policy, execution assumptions, and cost
assumptions.

## Risk

If `StrategyPortfolioSpec` keeps carrying non-allocation settings, alpha-os can
blur:

- strategy definition
- portfolio allocation policy
- evaluation assumptions
- runtime portfolio state

That makes it harder to reason about which part of a strategy decides target
weights and which part describes how those weights are evaluated or realized.

## Boundary

This is separate from `strategy-portfolio-default-boundary.md`.

That issue is about repeated default-like portfolio settings.

This issue is about the responsibility boundary of `StrategyPortfolioSpec`.

The intended direction is:

```text
Portfolio allocation:
  - portfolio construction
  - selection

Not portfolio allocation:
  - execution policy
  - holding cost assumptions
  - rebalance friction assumptions
```

This follows [`portfolio-allocation-boundary.md`](../design/portfolio-allocation-boundary.md):
do not add a larger generic portfolio policy object. Prefer explicit strategy
components with narrow responsibilities.

## Close Condition

Close this when `StrategyPortfolioSpec` no longer owns execution, holding-cost,
or rebalance-friction assumptions, or when those exceptions are explicitly
documented as intentional.

The minimum acceptable end state is:

- the allocation-related fields are distinguishable from execution and cost
  assumptions
- `TradingStrategySpec.portfolio` is not the only place to understand strategy
  execution and cost behavior
- a future allocator implementation can depend on allocation inputs without
  pulling in execution or cost assumptions
