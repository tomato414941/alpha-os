# Strategy Portfolio Naming Boundary

## Problem

In trading, a portfolio usually means the holdings or allocation at a point in
time.

`TradingStrategySpec.portfolio` does not represent current holdings. It
represents policy-like configuration:

- portfolio construction
- selection
- sizing
- rebalance behavior
- risk constraints
- execution assumptions
- holding cost assumptions

The name can make a strategy spec look like it owns a live or current portfolio,
when it actually owns configuration for constructing and evaluating portfolio
decisions.

## Risk

If `strategy.portfolio` keeps being read as a current allocation, alpha-os can
blur:

- strategy definition
- portfolio construction policy
- evaluation assumptions
- runtime portfolio state

That makes it harder to reason about what a strategy spec is allowed to contain.

## Boundary

This is separate from `strategy-portfolio-default-boundary.md`.

That issue is about repeated default-like portfolio settings.

This issue is about whether the field name `portfolio` is too broad or
misleading for the data it stores.

## Close Condition

Close this when the strategy schema has a clear name or documented convention
that distinguishes portfolio construction policy from an actual portfolio
holding state.
