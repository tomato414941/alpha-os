# Equal-weight long-only allocator adoption boundary

## Issue

`EqualWeightLongOnlyAllocator` exists as a narrower alternative to the rich
portfolio sizing path.

The underlying problem remains: a simple long-only candidate should not need to
depend on the full `portfolio_sizing_policy.py` request machinery. However,
connecting the allocator before controlling `portfolio_sizing_policy.py` risks
creating unused or parallel evaluation paths.

## Current State

- `EqualWeightLongOnlyAllocator` is implemented as an isolated component.
- `portfolio_sizing_policy.py` is still the active rich sizing path.
- Simple equal-weight behavior still exists inside the rich path as fallback
  logic.

## Acceptance Criteria

Before wiring `EqualWeightLongOnlyAllocator` into an evaluation path:

- The active callers of `portfolio_sizing_policy.py` are inventoried.
- The policy IDs that must remain supported are known.
- A simple long-only equal-weight candidate has one chosen evaluation path.
- Wiring the allocator either replaces an existing path or removes an existing
  branch; it must not only add a parallel route.
