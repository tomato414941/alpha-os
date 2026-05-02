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

### Direction

`direction_mode` is allocation-layer behavior. It controls how candidate target
weights are filtered: `long_short` keeps both signs, `long_only` keeps positive
targets, and `short_only` keeps negative targets.

`long_only` should be treated as a legacy or derived compatibility flag, not as a
separate source of truth from `direction_mode`.

`EqualWeightLongOnlyAllocator` should be understood as a long-only allocation
policy, not as a separate direction system.

### Rebalance

`rebalance_interval_steps` is not allocator internals. It controls when an
allocator is invoked, not how target weights are produced at a point in time.

An allocator should transform the current position candidates into current
target weights. It should not own scheduling, state retention, or the decision
to keep prior weights between rebalance dates.

Rebalance cadence should be treated as strategy cadence or evaluation execution
policy, depending on whether the cadence is part of the trading hypothesis or an
evaluation override.

### Risk And Exposure Constraints

Risk and exposure constraints are not allocator internals. They belong after raw
target weights are produced.

The existing `portfolio_construction_pipeline.py` is close to this role: it
takes target weights and applies direction filtering, overlays, top-k filtering,
group caps, risk-budget normalization, target-vol scaling, gross exposure caps,
and net exposure targeting.

The problem is not that constraints exist. The problem is that
`PortfolioConstructionSpec` mixes allocation policy, constraint policy,
rebalance cadence, report contract fields, and legacy compatibility in one
object.

`EqualWeightLongOnlyAllocator` may keep a minimal gross exposure cap for a simple
standalone rule, but richer constraints such as target volatility, leverage
caps, net exposure targets, group caps, and risk budgets should remain outside
the allocator itself.

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
