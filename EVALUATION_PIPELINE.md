# Evaluation Pipeline Design

## Principle

Individual signal quality is measured by **IC** (prediction accuracy).
Portfolio quality is measured by **Sharpe** (profitability after costs).
These are never mixed: IC is per-signal, Sharpe is per-portfolio.

## Prediction Targets

The system should predict multiple targets, not just returns. Different
targets capture different edges and diversify the portfolio's source of
profit. Each signal carries metadata about what it predicts and at what
horizon.

### Target types

| Target | Definition | Use |
|--------|-----------|-----|
| Residualized return | asset_return - benchmark_return | Directional alpha |
| Volatility | future realized vol vs current | Position sizing, vol targeting |
| Cross-sectional rank | which assets outperform others | Market-neutral strategies |

Residualized returns are the starting point. Volatility and cross-sectional
targets are natural extensions using the same IC evaluation framework.

### Residualization

All return-based evaluation uses **residualized returns** — benchmark-subtracted.
Raw return prediction conflates market direction with alpha.

```
residual_return[t] = asset_return[t] - benchmark_return[t]
```

Benchmark: equal-weight portfolio of broad assets (configured in default.toml).

### Horizons

Each signal is evaluated at multiple horizons. The best horizon is stored
as metadata — it is a property of the signal, not a system-wide setting.

```
horizons = [1, 5, 20]  (days)

fwd_return_h[t] = prices[t+h] / prices[t] - 1
residual_h[t]   = fwd_return_h[t] - benchmark_h[t]
IC_h             = spearmanr(signal, residual_h)

best_horizon     = argmax(IC_h)
```

A signal that predicts 5-day residual returns but not 1-day returns is
valuable. Without multi-horizon evaluation, it would be discarded.

### Signal metadata

Each signal carries:
- **target**: what it predicts (residual_return, volatility, rank)
- **horizon**: at what time scale (1, 5, 20 days)
- **per-asset IC**: prediction accuracy per asset class

These are properties of the signal discovered during evaluation, not
system-wide settings.

## Eval Universe

Evaluation asset count is a tradeoff: more assets per signal vs more
signals evaluated. The pipeline uses different depths at different stages.

**Generator (speed priority)**: 20-50 assets, balanced across asset classes.
More candidates explored is more valuable than more assets per candidate.

```
crypto: 5, US stocks: 10, ETFs: 5, commodities: 5, ...
```

**Admission (precision priority)**: all ~919 OHLCV assets.
Adoption is a one-time decision — thoroughness matters. Per-asset IC is
stored, providing the data for trading universe selection.

```
(sub (abs fear_greed) (delta_30 oil_wti))
  crypto:      IC = +0.03  (12 assets)
  US stocks:   IC = +0.01  (200 assets)
  ETFs:        IC = -0.005 (30 assets)
  commodities: IC = +0.04  (15 assets)
  overall:     IC = +0.02
```

The Generator eval universe is a fixed set cached to disk
(`data/eval_universe.json`). Selected once via correlation clustering
with explicit asset-class balancing. Recomputed only when explicitly
requested.

## Pipeline Stages

### 1. Generate

DSL/GP (or ML) produces candidate signals. Thousands per round.

### 2. Evaluate (IC)

Each candidate is evaluated against the eval universe:

```
for each horizon h in [1, 5, 20]:
    for each asset in eval_universe:
        IC_h[asset] = spearmanr(signal, residual_return_h[asset])
    mean_IC_h = mean(IC_h)
best_h = argmax(mean_IC_h)
fitness = mean_IC[best_h]
```

Stored in discovery pool: `(expression, fitness, horizon, behavior_vector)`.

This is a fast screen. No backtest, no cost model.

### 3. Validate (OOS IC)

Admission validates that IC is robust out-of-sample:

- Purged walk-forward CV: compute OOS IC per fold at the stored horizon
- IC must be consistently positive across folds
- PBO (Probability of Backtest Overfitting) gate
- DSR (Deflated Sharpe Ratio) gate applied to IC-derived returns

Admission does NOT run backtests or compute Sharpe for individual signals.

### 4. Combine (Portfolio)

Admitted signals are combined into a portfolio signal:

- Current: TC (True Contribution) weighting — linear
- Future: ML combiner — can learn interactions, time-varying weights

**Portfolio-level evaluation uses Sharpe, drawdown, costs.**
This is the only place Sharpe appears in the pipeline.

### 5. Trade

The combined signal is executed. Each underlying alpha's horizon
determines rebalancing frequency:

- horizon=1: daily rebalance
- horizon=5: rebalance every 5 days
- horizon=20: rebalance every 20 days

Mixed horizons in the portfolio provide natural time-scale diversification.

## Signal Generator Interface

DSL expressions and ML models are the same abstraction:

```
Input:  data: dict[str, np.ndarray]  (features)
Output: signal: np.ndarray           (one value per day)
```

Differences in lifecycle (ML needs retraining, serialization is different)
are handled by the generator, not the pipeline. The pipeline only sees
signals.

## What This Eliminates

- `fitness_metric` config — gone. IC is always used for signals, Sharpe for portfolio.
- IC vs Sharpe confusion — each has exactly one role.
- Backtest in Generator — unnecessary. IC is sufficient as a fast screen.
- Backtest in Admission — replaced by OOS IC validation.
- Raw returns in evaluation — always residualized.
- Per-call eval universe — fixed and cached.

## Implementation Order

1. Multi-horizon forward returns in `evaluate_cross_asset`
2. Residualized returns in IC path (use existing benchmark infrastructure)
3. Store horizon in discovery pool schema
4. Rewrite admission to validate OOS IC (not backtest Sharpe)
5. Remove `fitness_metric` config, hardcode IC for signals
6. Portfolio-level Sharpe evaluation (separate from signal evaluation)

## Constraints

- Memory: 7.6GB server limits concurrent data loading
- eval universe cached to avoid recomputation
- IC computation is cheap (rank correlation), enabling high throughput
- ML integration is future work; pipeline designed to accommodate it
