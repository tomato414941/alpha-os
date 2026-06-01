# Decision Backtest Rollout Boundary

## Problem

`decision_backtest.py` currently behaves like a portfolio-decision rollout
adapter, not a general trading-strategy backtest engine.

It builds `PortfolioDecisionInput` from `signal_series`, calls
`strategy.decide(...)`, treats the returned portfolio targets as desired
positions, and computes portfolio state transitions, turnover, costs, exposure,
PnL, and drawdown.

That contains useful rollout accounting, but the boundary is too specific:

- input is centered on `signal_series`
- strategy invocation is coupled to `PortfolioDecisionInput`
- output is coupled to `PortfolioDecisionOutput.targets`
- the name `decision_backtest` does not say whether this is a strategy rollout,
  portfolio rollout, or adapter around the current portfolio-decision types

## Direction

Keep the useful core:

```text
strategy output -> portfolio state transition -> realized PnL / cost / drawdown
```

Do not treat the current file as the source of truth for the universal
`TradingStrategy` contract.

The current `PortfolioDecisionInput` / `PortfolioDecisionOutput` path should be
understood as one adapter path only. A future rollout should be able to evaluate
other concrete trading strategies without forcing them through this exact I/O
shape.

## Non-Goal

Do not introduce a fixed strategy-rollout framework just to replace
`decision_backtest.py`.

The next shape may be a small function, an adapter, or a narrow module. It
should be driven by the concrete strategy and environment being evaluated.

## Close Condition

Close this when the rollout accounting / portfolio state transition core is
separated from the current signal-series and portfolio-decision adapter
assumptions, or when the file is deleted because a simpler strategy rollout path
replaces it.
