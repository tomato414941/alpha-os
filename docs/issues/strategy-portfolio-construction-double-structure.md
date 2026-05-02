# Strategy portfolio construction double structure

## Issue

`TradingStrategySpec.portfolio` currently contains `StrategyPortfolioSpec`, and
`StrategyPortfolioSpec` contains `PortfolioConstructionSpec`.

This creates a double structure:

```text
TradingStrategySpec
  -> StrategyPortfolioSpec
      -> PortfolioConstructionSpec
```

The boundary between the two is not clear enough. For example,
`selection_kind` is stored on `StrategyPortfolioSpec`, while `top_k` is stored on
`PortfolioConstructionSpec`, even though they are closely related.

This makes small allocation components such as `EqualWeightLongOnlyAllocator`
hard to connect without adding another parallel path.

## Current Suspects

- `selection_kind` vs `top_k`
- `sizing_policy`
- `rebalance_interval_steps`
- `long_only` / `direction_mode`
- exposure and risk constraints
- execution and holding-cost assumptions that may not belong to a strategy spec
- legacy document compatibility through `from_legacy()`

## Acceptance Criteria

- A field mapping exists between `StrategyPortfolioSpec` and
  `PortfolioConstructionSpec`.
- Each mapped field is classified as one of:
  - strategy-owned
  - evaluation/backtest-owned
  - execution-owned
  - legacy compatibility
- Closely related fields are assigned to one layer, not split across both.
- One layer is chosen as the future source of truth.
- The other layer is either marked legacy/adapter-only or given a narrower
  responsibility.

