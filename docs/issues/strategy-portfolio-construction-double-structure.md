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

## Current Decision

### Selection

`selection_kind` and `top_k` belong together. `top_k` is a parameter of the
`top_k` selection mode, not an independent portfolio construction concern.

Selection should be treated as part of the portfolio allocation layer: it decides
which position candidates are eligible to receive weights before sizing assigns
the final target weights.

`StrategyPortfolioSpec` should not split a selection mode from its parameters.

### Sizing

Existing `sizing_method=equal_weight` is not the same concept as
`EqualWeightLongOnlyAllocator`.

The existing `sizing_method` field is part of the rich sizing path. It also
implies backend classification such as `sizing_engine`, `sizing_family`, history
requirements, optimizer/report labeling, and skfolio-style model selection.

`EqualWeightLongOnlyAllocator` should not be wired in as a replacement for
`PortfolioConstructionSizingSpec.sizing_method`.

Sizing should eventually become an internal detail of the portfolio allocation
layer. Externally, a strategy should describe the allocation policy it wants,
while the allocation layer decides whether that policy is implemented by a
simple rule, a history-based allocator, or an optimizer.

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
