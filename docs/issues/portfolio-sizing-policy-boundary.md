# Portfolio Sizing Policy Boundary

## Problem

`portfolio_sizing_policy.py` is still on the current evaluation path, but it
carries too many responsibilities in one place.

It currently includes:

- sizing policy dataclasses
- allocator dispatch
- signal-weighted sizing
- constrained optimizer sizing
- signed mean-variance sizing
- historical model sizing
- cvxpy backend logic
- fallback equal-weight allocation
- risk, cost, uncertainty, and output helpers

The file also used to define `PortfolioAllocator`, while
`portfolio_allocation.py` now defines smaller allocation-oriented allocators such
as `EqualWeightLongOnlyAllocator`.

## Risk

The word allocator can start meaning two different things:

- converting position candidates into target weights
- converting a rich sizing request into a full sizing solution

If those meanings stay mixed, alpha-os can recreate the same portfolio boundary
confusion in code even after the strategy spec is cleaned up.

## Boundary

This issue does not assume `portfolio_sizing_policy.py` must stay on every
evaluation path.

The current sizing path is used by backtests, decision services, and signal
discovery evaluation. The issue is that its name and contents no longer make the
boundary obvious.

Simple long-only candidates should not have to pass through the full rich sizing
request machinery when a narrower path is enough.

The first cleanup should be small:

- keep the existing rich-request allocator name explicit as
  `PortfolioSizingAllocator`
- keep the smaller `portfolio_allocation.py` allocator boundary focused on
  position candidates to target weights
- avoid introducing a large generic portfolio allocation schema
- let simple candidate backtests bypass the rich sizing path when they only need
  long/flat equal-weight allocation

## Close Condition

Close this when the sizing path and allocation path have distinct names and
responsibilities.

The minimum acceptable end state is:

- `PortfolioSizingAllocator` names the rich sizing-request layer
- `portfolio_sizing_policy.py` may remain for rich sizing, but simple candidates
  are not forced through it
- small allocation implementations can be reused without pulling in the full
  sizing request machinery
