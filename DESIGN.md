# Alpha-OS Design Notes

## Core Philosophy

Alpha factors are not permanent — they are adaptive organisms.
No alpha works across all market regimes. The system's strength lies not in
finding a "golden alpha" but in continuously generating, deploying, and
retiring alphas as conditions change.

## Architecture: Generate → Gate → Monitor → Retire

```
Generate (GP/random)  →  Adoption Gate  →  Forward Test  →  Retire
     ↑                                         |              |
     └─────────── continuous cycle ─────────────┘              │
                                                               ↓
                                                          (learnings)
```

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

**Empirical evidence (2025-02-25)**:
- 5,792 candidate alphas on 2,084 days of BTC data (2020-06 ~ 2026-02).
- Walk-Forward CV: best OOS Sharpe ≈ 1.1, all 5 folds positive.
- Deflated Sharpe Ratio: ALL failed (p = 1.0) after multiple-testing correction
  with n_trials = 5,792.
- Interpretation: the gate is correctly rejecting data-mined results, but it
  is also impossible to pass because no single expression works across COVID
  crash, 2021 bull run, 2022 bear, and 2024 ETF rally simultaneously.

## Design Direction: Window-Based Adoption

Replace "full-history gate" with "recent-window gate":

| Current                           | Proposed                              |
| --------------------------------- | ------------------------------------- |
| Backtest on all available data    | Backtest on trailing N days (e.g. 200)|
| DSR n_trials = all candidates     | DSR n_trials = candidates per cycle   |
| PBO on full history               | PBO on recent window                  |
| Adopt rarely, keep forever        | Adopt frequently, retire quickly      |

The Forward Test lifecycle (ACTIVE → PROBATION → RETIRED) already handles
quality control post-adoption. Trust the exit mechanism; loosen the entrance.

### Benefits

- Alphas adapted to current regime pass the gate.
- Higher portfolio turnover, but each alpha is regime-appropriate.
- DSR becomes passable because the effective trial count per window is smaller.
- Aligns with biological analogy: short-lived, fast-reproducing organisms
  adapt better than long-lived rigid ones.

### Risks

- Higher transaction costs from frequent alpha rotation.
- Potential for adopting noise in low-volatility periods.
- Mitigation: keep Forward Test degradation window and cost-aware gate.

## Data Infrastructure

### Signal-Noise Integration

Alpha-OS consumes market data from the signal-noise API:

- Default: real data from `data/alpha_cache.db` (SQLite cache)
- API sync when signal-noise is running (localhost:8000)
- `--synthetic` flag for testing with random walks (clearly labeled)
- No silent fallback — missing data raises an error

### Current Data Coverage (alpha_cache.db)

| Signal       | Days  | Range               |
| ------------ | ----- | ------------------- |
| btc_ohlcv    | 2,084 | 2020-06 ~ 2026-02   |
| vix_close    | 9,129 | 1990-01 ~ 2026-02   |
| sp500        | 24,653| 1927-12 ~ 2026-02   |
| dxy          | 14,004| 1971-01 ~ 2026-02   |
| fear_greed   | 2,943 | 2018-02 ~ 2026-02   |
| gold         | 6,395 | 2000-08 ~ 2026-02   |
| tsy_yield_*  | 6,287 | 2001-01 ~ 2026-02   |

BTC constrains the intersection to ~2,084 days.
