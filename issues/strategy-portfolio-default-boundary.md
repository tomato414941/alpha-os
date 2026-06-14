# Strategy Portfolio Default Boundary

## Problem

Some hypotheses do not primarily test portfolio construction, but their strategy
documents still repeat full portfolio settings.

For example, the `crypto_regime_momentum` hypothesis is mainly about a candidate
rule. Its portfolio settings are fixed comparison assumptions:

- all assets
- equal weight
- daily rebalance
- long only
- gross exposure cap 1.0

These settings look like a default portfolio profile, but they are currently
written directly into each strategy document.

## Risk

If every strategy document repeats default-like portfolio settings, the boundary
between the strategy idea and the fixed evaluation portfolio becomes unclear.

That can make a hypothesis look more complex than it is, and can encourage
`TradingStrategySpec` to keep growing.

## Next Decision

Decide whether simple fixed portfolio assumptions should remain inline in each
strategy document or be represented by a small reusable portfolio profile.

## Close Condition

Close this when alpha-os has a clear rule for when portfolio construction is
part of the strategy being tested and when it is only a fixed default comparison
assumption.
