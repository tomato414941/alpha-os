# Alpha-OS Design Notes

## Core Philosophy

Alpha factors are not permanent — they are adaptive genomes in an ecosystem.
No alpha works across all market regimes. The system's strength lies not in
finding a "golden alpha" but in continuously generating, deploying, and
demoting alphas as conditions change.

### Biological Metaphor

Each alpha is a genome — not an individual organism, but a complete set of
instructions that defines a strategy. The full set of alphas forms an
ecosystem of coexisting genomes that compete for influence.

- **Genome** = one alpha's DSL expression. A self-contained blueprint, not
  a living thing. Crossover recombines two genomes into offspring.
- **Ecosystem** = all adopted genomes. The portfolio is the aggregate
  output of the ecosystem, not any single genome.
- **Niche** = a region of the strategy space (MAP-Elites). Each niche
  holds the genome best adapted to it.
- **Population** = diversity-weighted influence. A strong genome has many
  individuals (high weight); a weakening genome has fewer. The weight is
  the population size of that genome in the ecosystem.
- **DORMANT** = near-extinct. The genome is preserved but its population
  is effectively zero, ready to recover if conditions return.

What is bounded is the **ecosystem carrying capacity** — the number of
genomes the system can evaluate and maintain.

## Architecture

```
  Evolve (GP + MAP-Elites)
      │
      ▼
  Validate (Purged WF-CV, DSR, PBO)
      │
      ▼
  Adoption Gate ──── fail ──→ REJECTED
      │
      pass
      ▼
  ACTIVE ◄───────── revival ──── DORMANT
      │                             ▲
      │ degraded                    │ severe
      ▼                             │
  PROBATION ── recovery ──→ ACTIVE  │
      │                             │
      └──── continued decline ──────┘
```

The pipeline runs as a continuous cycle: generate candidates, validate
statistically, adopt survivors, then monitor and demote as edges decay.

## Alpha Expression DSL

Alphas are expressed as S-expression trees over a domain-specific language.
Each expression maps a dictionary of time-series signals to a single output
signal (numpy array).

### Node Types (8)

| Type            | Syntax                          | Example                                       |
| --------------- | ------------------------------- | --------------------------------------------- |
| Feature         | `name`                          | `btc_ohlcv`                                   |
| Constant        | `value`                         | `0.5`                                          |
| UnaryOp         | `(op child)`                    | `(neg btc_ohlcv)`                              |
| BinaryOp        | `(op left right)`               | `(sub nvda sp500)`                             |
| RollingOp       | `(op_window child)`             | `(mean_20 btc_ohlcv)`                         |
| PairRollingOp   | `(op_window left right)`        | `(corr_60 btc_ohlcv sp500)`                   |
| ConditionalOp   | `(op cond_l cond_r then else)`  | `(if_gt vix_close 30.0 (neg btc) (roc_10 btc))` |
| LagOp           | `(op_window child)`             | `(lag_5 btc_ohlcv)`                           |

### Operators

- **Unary**: neg, abs, sign, log, zscore
- **Binary**: add, sub, mul, div, max, min
- **Rolling**: mean, std, ts_max, ts_min, delta, roc, rank, ema
- **PairRolling**: corr, cov
- **Conditional**: if_gt (element-wise: cond_l > cond_r ? then : else)
- **Lag**: lag (returns x[t-N], first N values are NaN)

### Windows

Allowed window sizes: 5, 10, 20, 30, 60 days.

### Evolution

Expressions evolve via GP (population=200, generations=30):
- **Crossover** (50%): subtree swap between two parents
- **Mutation** (30%): swap feature, change window, or replace operator
- **Selection**: tournament (size=3) with elitism
- **Bloat control**: fitness penalty of 0.01 × node_count, max depth=3
- **Quality-diversity**: MAP-Elites archive with 4D behavior descriptor
  (correlation, holding half-life, turnover, complexity)

## Key Insight: Adaptive Market Hypothesis

- **Arbitrage decays**: once a pattern is discovered, participants exploit it
  and the edge disappears.
- **Regime change**: monetary policy, technology, and market structure shift
  the rules.
- **Markets are evolutionary, not efficient**: alpha is a temporary ecological
  niche that competition eventually fills.

### Implication for Validation

The original Adoption Gate required all-history backtesting (OOS Sharpe ≥ 0.5,
PBO ≤ 0.50, DSR p ≤ 0.05 across the full dataset). This implicitly assumes
time-invariant alpha — which contradicts the adaptive hypothesis.

**Empirical evidence (2025-02-25, with strict gates)**:
- 5,792 candidate alphas on 2,084 days of BTC data (2020-06 ~ 2026-02).
- Walk-Forward CV: best OOS Sharpe ≈ 1.1, all 5 folds positive.
- Deflated Sharpe Ratio: ALL failed (p = 1.0) after multiple-testing correction
  with n_trials = 5,792.
- Interpretation: the gate is correctly rejecting data-mined results, but it
  is also impossible to pass because no single expression works across COVID
  crash, 2021 bull run, 2022 bear, and 2024 ETF rally simultaneously.

**Current state (2026-02-26, with relaxed gates)**:
- 22,625 candidate alphas generated; 22,622 active, 3 rejected.
- 456 daily signals from signal-noise database.
- PBO and DSR thresholds relaxed to ≤ 1.0 (always pass); lifecycle manages
  quality post-adoption instead.

## Window-Based Adoption

Replace "full-history gate" with "recent-window gate":

| Before                            | After                                 |
| --------------------------------- | ------------------------------------- |
| Backtest on all available data    | Backtest on trailing N days (e.g. 200)|
| DSR n_trials = all candidates     | DSR n_trials = candidates per cycle   |
| PBO on full history               | PBO on recent window                  |
| Adopt rarely, keep forever        | Adopt frequently, demote quickly      |

Trust the exit mechanism (lifecycle); loosen the entrance (gate).

### Benefits

- Alphas adapted to current regime pass the gate.
- Higher portfolio turnover, but each alpha is regime-appropriate.
- DSR becomes passable because the effective trial count per window is smaller.

### Risks

- Higher transaction costs from frequent alpha rotation.
- Potential for adopting noise in low-volatility periods.
- Mitigation: keep Forward Test degradation window and cost-aware gate.

## Alpha Lifecycle

Five states with automatic transitions based on rolling Sharpe (63-day window):

| Transition             | Condition                    |
| ---------------------- | ---------------------------- |
| BORN → ACTIVE          | Passes adoption gate         |
| BORN → REJECTED        | Fails adoption gate          |
| ACTIVE → PROBATION     | Rolling Sharpe < 0.3         |
| PROBATION → ACTIVE     | Rolling Sharpe ≥ 0.5         |
| PROBATION → DORMANT    | Rolling Sharpe < 0           |
| DORMANT → PROBATION    | Rolling Sharpe ≥ 0.3         |

DORMANT alphas are not traded but remain monitored. If market conditions
shift and their rolling Sharpe recovers, they re-enter PROBATION and can
return to ACTIVE. This avoids permanently discarding alphas that may have
regime-specific value.

## Adoption Gate

All criteria must pass for an alpha to be adopted:

| Check         | Threshold | Notes                             |
| ------------- | --------- | --------------------------------- |
| OOS Sharpe    | ≥ 0.5     | Purged Walk-Forward CV (5 folds)  |
| PBO           | ≤ 1.0     | Batch-computed, warn only         |
| DSR p-value   | ≤ 1.0     | Disabled — lifecycle manages      |
| Correlation   | ≤ 0.5     | Avg correlation with live alphas  |
| Min days      | ≥ 200     | Minimum data requirement          |

PBO and DSR thresholds are intentionally relaxed (≤ 1.0 = always pass).
The lifecycle system handles quality control post-adoption: alphas that
degrade are demoted through PROBATION → DORMANT → eventually pruned.

## Signal Combination: Quality × Diversity Weighting

All active and probation alphas participate in the combined signal.
Each alpha's weight reflects both its recent performance (quality) and
its uniqueness relative to the portfolio (diversity):

```
weight_i = max(rolling_sharpe_i, 0) × diversity_i + min_weight
```

Where:
- **Quality** = `max(rolling_sharpe, 0)` — negative Sharpe alphas get
  minimal weight but are not excluded (min_weight floor).
- **Diversity** = `1 - mean(|corr(i, j)|)` for all j ≠ i — alphas
  uncorrelated with the rest contribute more.
- **min_weight** = 1e-4 — ensures no alpha is fully zeroed out.

Diversity scores are recomputed every 63 days using a 252-day lookback
window. Correlation is computed via chunked matrix multiplication
(`chunk_size=1000`) to keep memory bounded for large alpha pools.

This replaces the earlier equal-weight combiner which selected a fixed
number of low-correlation alphas and weighted them equally.

## Paper Trading

Two modes:

- **Daily cycle** (`paper --once`): Syncs data, evaluates all alphas,
  applies lifecycle transitions, combines signals with quality × diversity
  weights, adjusts for risk (drawdown + volatility scaling), and executes
  via paper executor.
- **Backfill simulation** (`paper --backfill`): Vectorized historical
  simulation. Pre-computes all signal matrices once, then iterates days
  via array indexing. Orders of magnitude faster than per-day evaluation.

Risk adjustment applies three-stage drawdown scaling (5% / 10% / 15%)
and volatility targeting (15% annualized).

## Data Infrastructure

### Signal-Noise Integration

Alpha-OS dynamically discovers all daily signals from the signal-noise
database (456 signals as of 2026-02-26). Signals are bulk-imported into a local SQLite
cache (`data/alpha_cache.db`) on each run.

- Default: real data from signal-noise DB + local cache
- API sync when signal-noise is running (localhost:8000)
- `--synthetic` flag for testing with random walks (clearly labeled)
- No silent fallback — missing data raises an error

### Data Coverage (alpha_cache.db)

| Signal          | Days   | Range               |
| --------------- | ------ | ------------------- |
| btc_ohlcv       | 2,084  | 2020-06 ~ 2026-02   |
| vix_close       | 9,129  | 1990-01 ~ 2026-02   |
| sp500           | 24,653 | 1927-12 ~ 2026-02   |
| dxy             | 14,004 | 1971-01 ~ 2026-02   |
| fear_greed      | 2,943  | 2018-02 ~ 2026-02   |
| gold            | 6,395  | 2000-08 ~ 2026-02   |
| tsy_yield_10y/2y| 6,288  | 2001-01 ~ 2026-02   |
| tsy_yield_*     | 1,033  | 2022-01 ~ 2026-02   |

BTC constrains the intersection to ~2,084 days.
