# Portfolio Allocation Boundary

Do not modify this file unless the user explicitly asks to change
`portfolio-allocation-boundary.md`.

## Decision

Do not model portfolio allocation as one generic configurable object that can do
everything.

Prefer small allocator implementations with narrow contracts.

An allocator owns its policy internally:

```text
Position candidates + minimal context
  -> PortfolioAllocator implementation
  -> target weights
```

Examples:

- `EqualWeightLongOnlyAllocator`
- `InverseVolAllocator`
- `SkfolioMeanRiskAllocator`

Each implementation may use hand-written logic or an external library such as
`skfolio`, `PyPortfolioOpt`, `Riskfolio-Lib`, or `cvxpy`. The caller should not
need to know how the weights were produced.

## Why

A large generic allocation config tends to become another manifest-shaped
surface:

```text
policy + constraints + optimizer config + fallback config + data config + ...
```

That increases schema size before the system knows which allocation ideas are
actually useful.

Small allocator classes keep unused policy out of the shared contract. Equal
weight should not need the inputs required by a covariance optimizer. A
skfolio-backed allocator should not force every other allocator to expose
skfolio-shaped configuration.

## Minimal Contract

The common boundary should stay narrow:

```text
allocate(context) -> target weights
```

The first context can be as small as:

```text
position candidates
```

An allocator that needs more information can define and require its own richer
context, such as a returns window.

## Non-Goals

This boundary is not:

- an execution engine
- an order-management layer
- a cost model
- a live portfolio state store
- a universal optimizer schema

Execution assumptions, holding costs, and broker behavior should not be hidden
inside the base portfolio allocation contract.

## Current Implication

`TradingStrategySpec.portfolio` currently carries more than a clean portfolio
allocation layer should carry. Future cleanup should move toward explicit
strategy components rather than adding a larger generic portfolio policy object.

