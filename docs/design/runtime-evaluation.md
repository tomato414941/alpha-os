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
- what horizon that truth implies
- what scoring semantics are valid

Hypotheses are not required to predict every target. Many hypotheses will only
be meaningful for a narrow subset.

So the intended relation is:

- `target`: what is predicted
- `hypothesis`: how it is predicted
- `hypothesis -> target binding`: which targets a hypothesis is allowed to predict

Evaluation should close within one target at a time. Predictions,
observations, and metrics are only comparable when they share the same target.

### Residualization

Return-based evaluation should use residualized returns rather than raw market
direction.

```text
residual_return[t] = asset_return[t] - benchmark_return[t]
```

### Horizons

Horizon should be a first-class part of target definition.

The system should support multiple horizons with the same contract rather than
treating one horizon as permanent truth and the others as exceptions.

### Signal Metadata

Each evaluated signal should carry at least:

- target
- horizon
- per-asset or per-sleeve quality where relevant

## Evaluation Universe

Evaluation asset count is a tradeoff between breadth and depth.

- generator stage: smaller universe, speed priority
- admission stage: broader universe, precision priority

The point is not to evaluate every candidate everywhere at the earliest stage,
but to preserve enough breadth to avoid local search collapse.

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
