# Runtime And Evaluation

## Core Principles

1. **Prediction-first, metadata-enriched**
   - evaluation is based on predictions
   - metadata may help risk management, but should not define evaluation truth
2. **Adaptive and online**
   - new evidence should continuously update the runtime view
3. **IC for hypotheses, Sharpe for portfolio**
   - hypothesis scoring and portfolio scoring must stay separate
4. **Internal market logic**
   - better predictors gain more influence over time

## Prediction Targets

The system should predict multiple targets, not only returns.

Different targets capture different edges and may contribute differently to the
portfolio.

### Target Types

| Target | Definition | Use |
|--------|-----------|-----|
| Residualized return | asset return minus benchmark return | Directional alpha |
| Volatility | future realized vol vs current | Position sizing, vol targeting |
| Cross-sectional rank | which assets outperform others | Market-neutral strategies |

Residualized returns are the natural starting point, but they should not remain
the only target class forever.

### Broader Target Classes

Beyond market-facing returns, the design should also allow targets such as:

- volatility and regime state
- correlation and market structure
- liquidity and execution quality
- hypothesis health and decay risk
- allocation and portfolio construction relevance

These should not all be implemented at once. The point is to keep the target
model broad enough that future target classes do not require a new runtime
concept each time.

### Target-Centric Model

Targets are first-class objects.

A target defines:

- what realized outcome counts as truth
- what subject that truth belongs to
- what output form a prediction should take
- what scoring semantics are valid
- any required parameters such as horizon, window, event boundary, or benchmark

Hypotheses are not required to predict every target. Many hypotheses will only
be meaningful for a narrow subset.

So the intended relation is:

- `target`: what is predicted
- `hypothesis`: how it is predicted
- `hypothesis -> target binding`: which targets a hypothesis is allowed to predict

Evaluation should close within one target at a time. Predictions,
observations, and metrics are only comparable when they share the same target.

### Target Schema

Targets should not be modeled as horizon labels alone. The minimal design
should treat each target as a small definition object with these axes:

| Field | Meaning | Example |
|-------|---------|---------|
| `family` | what is being predicted | `residual_return`, `realized_vol`, `regime_state` |
| `observation_kind` | how truth is constructed | `fixed_horizon`, `event_window`, `state_label` |
| `subject_kind` | what unit is being predicted | `asset`, `pair`, `sleeve`, `portfolio` |
| `output_kind` | what form the prediction takes | `real_value`, `probability`, `rank`, `class_label` |
| `scoring_kind` | what scoring semantics apply | `corr_mmc`, `rank_ic`, `log_loss` |
| `params` | target-specific parameters | `horizon_days`, `window_days`, `benchmark_ref` |

In this model, horizon is common but not universal. Many return targets will
need `horizon_days`, but state, event, and threshold targets may not map cleanly
to one fixed horizon.

```json
{
  "family": "residual_return",
  "observation_kind": "fixed_horizon",
  "subject_kind": "asset",
  "output_kind": "real_value",
  "scoring_kind": "corr_mmc",
  "params": {
    "horizon_days": 3,
    "benchmark_ref": "btc_spot_beta"
  }
}
```

```json
{
  "family": "regime_state",
  "observation_kind": "state_label",
  "subject_kind": "asset",
  "output_kind": "class_label",
  "scoring_kind": "log_loss",
  "params": {
    "window_days": 10
  }
}
```

### Residualization

Return-based evaluation should use residualized returns rather than raw market
direction.

```text
residual_return[t] = asset_return[t] - benchmark_return[t]
```

### Horizons

Horizon should be a first-class parameter when the target family requires it.

The system should support multiple horizons with the same contract rather than
treating one horizon as permanent truth and the others as exceptions. But the
runtime should not assume every target is fixed-horizon by construction.

### Signal Metadata

Each evaluated signal should carry at least:

- `target`
- any target parameters needed for evaluation
- per-asset or per-sleeve metrics where relevant

## Evaluation Universe

Evaluation asset count is a tradeoff between breadth and depth.

- generator stage: smaller universe, speed priority
- admission stage: broader universe, precision priority

The point is not to evaluate every candidate everywhere at the earliest stage,
but to preserve enough breadth to avoid local search collapse.

## Portfolio Decision

Portfolio decision should be treated as a first-class layer.

It is not the same thing as a hypothesis, a target, a scoring rule, or a meta
prediction. Those layers answer:

- what is being predicted
- how it is predicted
- how it is scored
- how multiple predictions are aggregated

Portfolio decision answers a different question:

- given those predictive objects, what portfolio state should exist now

So the intended separation is:

- **meta prediction**
  - what is likely to happen
- **portfolio decision**
  - what portfolio state should be chosen
- **execution**
  - how that desired state becomes bounded orders

The portfolio layer should therefore output portfolio intents such as:

- target weights
- target position deltas
- entry or no-trade gates
- risk scaling decisions

It should not output raw orders as its primary object. Orders belong to
execution.

### Theory-Driven Requirements

Portfolio decision should be derived from portfolio requirements first, not
from whatever the current runtime happens to expose.

At minimum, the design should make these questions explicit:

- **objective**
  - expected return, risk-adjusted return, utility, or another portfolio goal
- **risk model**
  - variance, downside risk, drawdown, tail risk, or another notion of risk
- **dependence**
  - correlation or covariance across assets, targets, and predictive sleeves
- **cost model**
  - turnover, slippage, market impact, and no-trade regions
- **uncertainty**
  - confidence, estimation error, and robustness to unstable signals
- **constraints**
  - leverage, concentration, liquidity, turnover, and capital limits
- **time**
  - whether the decision is one-shot, rolling, or stateful across periods

These questions define what the portfolio layer must do. Runtime details should
follow from them, not the other way around.

### Inputs To The Decision Layer

The eventual decision layer should be able to consume more than one kind of
signal. Typical inputs include:

- expected return style signals
- risk or volatility signals
- confidence or uncertainty signals
- diversification or dependence signals
- execution quality or cost signals
- current portfolio state

Different targets may feed different parts of the decision. For example:

- return targets may influence direction and expected reward
- volatility targets may influence sizing and risk scaling
- directional targets may act as entry filters
- execution targets may gate or defer trades

This is why target expansion only makes sense once the portfolio decision layer
has a clear role for those targets.

## Portfolio Conversion

Good hypotheses do not automatically imply a good portfolio.

The runtime should distinguish between:

- many candidate hypotheses
- many active hypotheses
- many effective independent bets

The thing that matters is effective breadth rather than raw count.

So the intended design is:

- broad search upstream
- constrained active set downstream
- capital concentrated on the most useful independent bets

## Layering

The runtime should keep these concerns separate:

- **evaluation**
  - whether a hypothesis predicts a target well
- **selection**
  - whether it stays in the active candidate set
- **allocation**
  - how much capital influence it deserves
- **execution**
  - how that influence becomes bounded trades

These layers should interact, but they should not collapse into one score or
one state variable too early.

## Cost And Correlation

Cost and crowding should affect portfolio decisions before execution-only
diagnostics.

In practice this means:

- turnover-sensitive hypotheses should look worse before they reach execution
- highly correlated hypotheses should compete for capital rather than all
  surviving at equal influence

The purpose is not to eliminate every similar hypothesis. The purpose is to
avoid turning many similar signals into one oversized hidden bet.

## Pipeline Stages

1. **Generate**
   - produce candidate predictions
2. **Evaluate**
   - measure predictive quality against realized outcomes
3. **Admit**
   - filter out obvious noise
4. **Select**
   - determine which hypotheses remain active in the bounded runtime
5. **Trade**
   - downstream execution from portfolio decisions

These stages should remain conceptually separate even when a convenience
wrapper runs them as one flow.
