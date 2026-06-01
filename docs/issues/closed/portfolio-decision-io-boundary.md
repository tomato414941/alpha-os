# Portfolio decision I/O boundary

## Resolution

Closed by deleting the unused `PortfolioDecisionInput`,
`PortfolioDecisionOutput`, related portfolio-decision input helper types, and
`portfolio_decision_inputs.py`.

The remaining active code keeps `SubjectSet` and `PortfolioState` only. A
future strategy rollout or portfolio allocator can introduce concrete I/O when
there is an actual caller that needs it.

## Problem

`PortfolioDecisionInput` and `PortfolioDecisionOutput` existed before the
`TradingStrategy` contract was introduced.

They were used by the removed portfolio-decision backtest implementation, but
they should not define what a trading strategy is. A trading strategy should not
be forced to depend on portfolio-decision-specific I/O just because those types
already existed.

## Current Use

`PortfolioDecisionInput` carries portfolio state, observed inputs, assumptions,
and subject metadata.

`PortfolioDecisionOutput` carries target portfolio weights and sizing
diagnostics.

This looks closer to a portfolio allocation / sizing boundary than to a general
trading strategy contract.

## Risk

If these types become the default strategy I/O, strategy design becomes
subordinate to the current portfolio sizing implementation.

That can make non-portfolio-allocation strategies, execution-aware strategies,
or future RL-style policies harder to model.

## Direction

Do not treat `PortfolioDecisionInput` / `PortfolioDecisionOutput` as the
universal trading strategy I/O contract.

`TradingStrategy` is now a generic protocol. Concrete strategies should choose
the input and output types they actually need instead of inheriting placeholder
strategy I/O marker types.

Before deleting or renaming the portfolio decision I/O types, decide whether
they are:

- still useful as portfolio allocation / sizing I/O
- only an adapter around the current sizing implementation
- replaceable by narrower allocation request/result types

## Current Decision

The old portfolio-decision backtest adapter was removed.

These types do not define the universal `TradingStrategy` contract. The generic
`TradingStrategy` protocol remains the source of truth: each concrete engine may
choose the input and output types it actually needs.

## Close Condition

Close this when `PortfolioDecisionInput` and `PortfolioDecisionOutput` are
either justified as portfolio allocation-layer I/O or replaced by narrower
types.
