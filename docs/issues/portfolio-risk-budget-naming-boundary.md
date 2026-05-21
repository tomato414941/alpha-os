# Portfolio risk budget naming boundary

## Problem

`portfolio_construction.risk_budget` is too broad for its current behavior.

The name sounds like a general portfolio risk budgeting concept, but the
current object mostly controls post-sizing normalization of portfolio target
weights.

Current fields are:

- `risk_normalization_mode`
- `target_gross_exposure`
- `allow_releverage`

This is not a general risk budget object.

## Current Use

`risk_budget` currently supports one active normalization mode:

- `gross`: scale target weights toward `target_gross_exposure`

In checked-in runtime manifests, the object is only used with
`risk_normalization_mode: gross`. In practice, it behaves like gross exposure
normalization, not broad portfolio risk budgeting.

## Risk

The current name invites unrelated concepts to be added under one object, such
as:

- portfolio risk allocation
- gross exposure targets
- volatility targeting
- leverage limits
- diagnostic thresholds
- sleeve-level risk budgets

It also overlaps with nearby fields:

- `target_vol`
- `gross_exposure_cap`
- `gross_leverage_cap`

This makes it unclear whether the object is a target, a cap, a normalization
policy, or an evaluation diagnostic contract.

## Direction

Do not expand `risk_budget` into a general risk or portfolio objective object.

Prefer a narrower future name if the behavior is kept, such as:

- `portfolio_normalization`
- `exposure_normalization`
- `target_exposure_policy`

Avoid adding volatility targeting back under this object. If volatility
targeting is needed, represent it explicitly instead of hiding it behind a
generic risk budget name.

## Close Condition

Close this when `risk_budget` has either been renamed to a narrower
normalization or exposure concept, or removed in favor of explicit
`target_vol`, `gross_exposure_cap`, and `gross_leverage_cap` behavior.
