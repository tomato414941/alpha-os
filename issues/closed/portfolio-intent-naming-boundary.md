# Portfolio intent naming boundary

Status: Closed. `PortfolioIntentSpec` and persisted `portfolio_intent` were
removed; concentration controls now live directly on `PortfolioConstructionSpec`.

## Problem

`portfolio_intent` is too broad for its current behavior.

The name sounds like a general expression of the portfolio's purpose, but the
current fields are concrete concentration and diversification controls:

- `effective_n_floor`
- `top_gross_share_cap_n`
- `top_gross_share_cap`
- `concentration_min_abs_weight`

This is not a general strategy intent object.

## Current Use

`portfolio_intent` is used in two places:

- sizing logic, where `diversified_risk_budget` uses effective-N and top-gross
  share constraints to blend weights toward equal weight
- evaluation diagnostics, where concentration failures are reported when the
  realized portfolio violates those thresholds

So the current object behaves more like a portfolio concentration policy or
diversification constraint set.

## Risk

The word `intent` invites unrelated meanings to be added later, such as:

- research rationale
- portfolio objective
- risk appetite
- investment mandate
- diagnostic expectation

That would make the strategy config broader and less executable.

## Direction

Do not expand `portfolio_intent` into a general-purpose strategy or research
intent object.

Prefer a narrower future name such as:

- `portfolio_concentration_policy`
- `diversification_constraints`

Before renaming, decide whether the object is:

- a sizing-policy input
- a post-sizing concentration constraint
- an evaluation diagnostic threshold
- some combination that should be split

## Close Condition

Close this when `portfolio_intent` has either been renamed to a narrower
concentration/diversification concept or replaced by explicit fields in the
appropriate sizing, constraint, and diagnostic layers.
