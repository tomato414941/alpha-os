# Evaluation Pipeline Design

## Principles

**1. Output only.**
Signal generators are black boxes. The pipeline sees only predictions,
never internals. DSL expression, ML model, LLM, human — all the same.
Evaluation judges "you predicted X, did X happen?" Nothing else.

This eliminates: complexity penalties, feature cap checks, semantic
duplicate detection, expression structure inspection. If two generators
produce identical output, they compete on equal footing — one survives
naturally.

**2. Adaptive and online.**
Evaluation is continuous, not batch. Every day, every signal's latest
prediction is scored against realized outcomes. No periodic retraining
cycles or manual review gates. The system learns what works and what
doesn't, always, automatically.

**3. IC for signals, Sharpe for portfolio.**
Individual signal quality is measured by IC (prediction accuracy).
Portfolio quality is measured by Sharpe (profitability after costs).
These are never mixed.

**4. Internal market efficiency.**
Good predictors gain influence. Bad predictors lose influence and die.
No manual thresholds — selection pressure is continuous and automatic,
like a market where accurate participants accumulate capital and
inaccurate ones go bankrupt.

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
- Statistical checks (PBO, DSR) as sanity filters, not hard gates

Signals that pass enter the live pool with a **minimum stake**.

### 4. Live Selection (Internal Market)

Each live signal carries a virtual stake. Stake determines portfolio
weight — this replaces manual TC weighting and threshold-based lifecycle.

```
every day:
    for each live signal:
        prediction = signal's output yesterday
        actual     = realized residual return today
        score      = correlation(prediction, actual)

        if score > 0:  stake *= (1 + reward_rate)
        else:          stake *= (1 - penalty_rate)

    normalize stakes → portfolio weights
```

Properties:
- **New signals enter small.** They must prove themselves before gaining
  influence. No immediate large allocation.
- **Accurate signals grow.** Compounding stake → increasing weight.
  The best predictors dominate the portfolio naturally.
- **Inaccurate signals shrink and die.** No manual demotion rules.
  Stake → 0 is natural death. Slot freed for new candidates.
- **Marginal contribution matters.** A signal with IC > 0 but
  identical to existing signals adds no value — its predictions are
  already covered, so its live score won't improve the portfolio.
  It stagnates and eventually dies. This is TC without computing TC.

This is analogous to:
- Numerai: models stake NMR, earn/lose based on live correlation
- Polymarket: accurate predictors profit, inaccurate ones lose
- Real markets: informed participants accumulate capital

### 5. Trade

Portfolio weights from the stake system determine position sizes.
Each signal's horizon determines rebalancing frequency:

- horizon=1: daily rebalance
- horizon=5: rebalance every 5 days
- horizon=20: rebalance every 20 days

Mixed horizons provide natural time-scale diversification.

**Portfolio-level evaluation uses Sharpe, drawdown, costs.**
This is the only place Sharpe appears in the pipeline.

## Signal Generator Interface

The pipeline sees one thing:

```
Input:  data: dict[str, np.ndarray]  (features)
Output: signal: np.ndarray           (one value per day)
```

What happens inside is irrelevant. DSL expression, gradient-boosted tree,
neural net, hand-written rule, LLM-generated strategy — all produce a
signal array and are judged solely by that output.

Lifecycle differences (ML needs retraining, DSL is stateless) are the
generator's problem. The pipeline doesn't know and doesn't care.

## What This Eliminates

- `fitness_metric` config — gone. IC for signals, Sharpe for portfolio. Fixed.
- Backtest in Generator/Admission — IC is sufficient.
- Raw returns — always residualized.
- Per-call eval universe — fixed and cached.
- Manual lifecycle thresholds — replaced by stake-based natural selection.
- Active/dormant/rejected state machine — replaced by continuous stake.
- TC computation — emergent from stake dynamics.
- Feature cap checks, semantic duplicate detection, complexity penalties —
  output-only evaluation makes internal inspection unnecessary.
- Separate handling for DSL vs ML — the pipeline is generator-agnostic.

## Implementation Order

1. Multi-horizon forward returns in `evaluate_cross_asset`
2. Residualized returns in IC path (use existing benchmark infrastructure)
3. Store horizon in discovery pool schema
4. Rewrite admission to validate OOS IC (not backtest Sharpe)
5. Remove `fitness_metric` config, hardcode IC for signals
6. Stake-based live selection (replace state machine lifecycle)
7. Portfolio weights from stakes (replace TC weighting)
8. Portfolio-level Sharpe evaluation (separate from signal evaluation)

## Constraints

- Memory: 7.6GB server limits concurrent data loading
- eval universe cached to avoid recomputation
- IC computation is cheap (rank correlation), enabling high throughput
- ML integration is future work; pipeline designed to accommodate it
