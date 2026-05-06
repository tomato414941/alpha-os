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

The boundary between the two is not clear enough. `selection_kind` and `top_k`
now live together on `StrategyPortfolioSpec`, but the surrounding
`PortfolioConstructionSpec` still mixes multiple responsibilities.

This makes small allocation components such as `EqualWeightLongOnlyAllocator`
hard to connect without adding another parallel path.

## Current Suspects

- `selection_kind` vs `top_k`
- `sizing_policy`
- `rebalance_interval_steps`
- `long_only` / `direction_mode`
- exposure and risk constraints
- execution and holding-cost assumptions that may not belong to a strategy spec

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

The existing `sizing_method` field is part of the legacy rich sizing path. It
also implies backend classification such as `sizing_engine`, `sizing_family`,
history requirements, optimizer/report labeling, and skfolio-style model
selection.

`portfolio_sizing_policy.py` should be treated as a legacy rich sizing path, not
as the default home for new small allocation rules.

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

`long_only` should be treated only as a derived runtime flag when older internal
call paths still need a boolean. It must not be accepted as a persisted strategy
document field or treated as a second source of truth from `direction_mode`.

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

Current decision: rebalance cadence is strategy-owned for strategy specs.
`StrategyPortfolioSpec.rebalance_interval_steps` is the source of truth.
`RebalancePolicySpec` has been removed. `PortfolioConstructionSpec` must not
accept or emit `rebalance_interval_steps` as a persisted construction field.

### Risk And Exposure Constraints

Risk and exposure constraints are not allocator internals. They belong after raw
target weights are produced.

The existing `portfolio_construction_pipeline.py` is close to this role: it
takes target weights and applies direction filtering, overlays, top-k filtering,
group caps, risk-budget normalization, target-vol scaling, gross exposure caps,
and net exposure targeting.

The problem is not that constraints exist. The problem is that
`PortfolioConstructionSpec` mixes allocation policy, constraint policy, and
report contract fields in one object.

`EqualWeightLongOnlyAllocator` may keep a minimal gross exposure cap for a simple
standalone rule, but richer constraints such as target volatility, leverage
caps, net exposure targets, group caps, and risk budgets should remain outside
the allocator itself.

## Field Inventory

Current classification:

| Field | Current location | Classification | Reason |
|---|---|---|---|
| `portfolio_construction` | `StrategyPortfolioSpec` | legacy / unclear | Parent object mixing allocation, constraints, and report contract fields. |
| `selection_kind` | `StrategyPortfolioSpec` | portfolio allocation | Chooses which candidates may receive weights. Belongs with `top_k`. |
| `top_k` | `StrategyPortfolioSpec` | portfolio allocation | Parameter of `selection_kind=top_k`; no longer belongs to `PortfolioConstructionSpec`. |
| `sizing_policy` | `PortfolioConstructionSpec` | portfolio allocation / legacy unclear | Related to weight creation, but also carries optimizer labels and history requirements. |
| `rebalance_interval_steps` | `StrategyPortfolioSpec` | strategy-owned | Strategy cadence is part of the trading hypothesis; `PortfolioConstructionSpec` is not an input source. |
| `long_only` | `PortfolioConstructionSpec` | derived runtime field | Derivable from `direction_mode`; should not be a persisted source of truth. |
| `direction_mode` | `PortfolioConstructionSpec` | portfolio allocation | Determines how long/short/flat candidates are treated before weights are finalized. |
| `active_overlay` | `PortfolioConstructionSpec` | portfolio allocation / unclear | Appears to alter target weights, but the boundary is still broad. |
| `gross_exposure_cap` | `PortfolioConstructionSpec` | portfolio allocation / constraint | Constrains total target weight after allocation. |
| `target_vol` | `PortfolioConstructionSpec` | portfolio allocation / constraint | Scales weights toward a volatility target. |
| `gross_leverage_cap` | `PortfolioConstructionSpec` | portfolio allocation / constraint | Constrains leverage after weights are produced. |
| `net_exposure_target` | `PortfolioConstructionSpec` | portfolio allocation / constraint | Constrains net exposure after weights are produced. |
| `asset_class_weight_caps` | `PortfolioConstructionSpec` | portfolio allocation / constraint | Group cap on target weights. |
| `cluster_weight_caps` | `PortfolioConstructionSpec` | portfolio allocation / constraint | Group cap on target weights. |
| `portfolio_intent` | `PortfolioConstructionSpec` | legacy / unclear | Captures effective-N and concentration constraints, but the name is vague. |
| `risk_budget` | `PortfolioConstructionSpec` | portfolio allocation / constraint | Risk normalization and target exposure controls. |
| `sleeve_composition` | `PortfolioConstructionSpec` | strategy decision / portfolio allocation | Could be core hypothesis structure or allocation blending. |
| `rebalance_friction_policy` | `StrategyPortfolioSpec` | strategy decision / evaluation assumption | Strategy-owned if it controls trade decisions; evaluation-owned if it only models friction. |
| `execution_policy` | `StrategyPortfolioSpec` | evaluation assumption | `fee_bps`, `market_impact_bps`, and spread assumptions are used to calculate net results. |
| `holding_cost_policy` | `StrategyPortfolioSpec` | evaluation assumption / strategy input | Evaluation-owned when deducted from returns; strategy-owned only when used for decisions. |

## Acceptance Criteria

- A field mapping exists between `StrategyPortfolioSpec` and
  `PortfolioConstructionSpec`.
- Each mapped field is classified as one of:
  - strategy-owned
  - evaluation/backtest-owned
  - execution-owned
  - legacy/internal adapter
- Closely related fields are assigned to one layer, not split across both.
- One layer is chosen as the future source of truth.
- The other layer is either marked legacy/adapter-only or given a narrower
  responsibility.
