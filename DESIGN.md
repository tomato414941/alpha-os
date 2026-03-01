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

Alpha-OS dynamically discovers all signals from the signal-noise database
(753 signals across 124 collectors as of 2026-02-26). Signals are
bulk-imported into a local SQLite cache (`data/alpha_cache.db`) on each run.

- Default: real data from signal-noise DB + local cache
- API sync when signal-noise is running (localhost:8000)
- `--synthetic` flag for testing with random walks (clearly labeled)
- No silent fallback — missing data raises an error

### Signal Coverage

| Metric               | Count | Notes                              |
| -------------------- | ----- | ---------------------------------- |
| Total signals in DB  | 753   | From 124 collectors                |
| Current (Feb 2026)   | 556   | Actively updated                   |
| Usable for BTC alpha | 448   | After missing/empty filter         |
| Missing/broken       | ~70   | Stale collectors or API deprecated |

Signals span diverse frequencies and domains:

| Category          | Examples                          | Frequency   | Count |
| ----------------- | --------------------------------- | ----------- | ----- |
| Crypto on-chain   | btc_ohlcv, defi_tvl_*, bc_*       | Daily       | ~50   |
| Traditional macro | sp500, vix_close, dxy, gold       | Daily       | ~30   |
| DeFi protocols    | defi_proto_*, defi_sc_*           | Daily       | ~30   |
| Futures (COT)     | cot_gold_net_c, cot_btc_oi        | Weekly      | ~39   |
| Economic (IMF)    | imf_gdp_growth_*, imf_inflation_* | Annual      | ~24   |
| Housing (BIS)     | bis_pp_*, oecd_hpi_*              | Quarterly   | ~60   |
| Consumer prices   | bls_cpi_*, ecb_hicp_*             | Monthly     | ~25   |
| Alternative       | wiki_*, gdelt_*, npm_*, so_*      | Daily       | ~120  |
| Climate/nature    | noaa_*, meteo_*, arctic_*         | Daily/Month | ~50   |

### Missing Data Handling

Non-daily signals (monthly, quarterly, annual) are handled transparently:

1. **`get_matrix()`** pivots all signals onto a common date index
2. **Forward-fill** (`ffill`) propagates the last known value forward
3. **Back-fill + zero-fill** handles leading NaN before a signal's first data point
4. **Price-gated rows**: only dates where the price signal (btc_ohlcv) exists are kept

This means a monthly CPI signal becomes a daily series where the value
stays constant until the next release — which is how the real world works.
Signals can appear or disappear without breaking the pipeline.

### BTC Data Window

BTC constrains the effective date range to ~2,089 days (2020-06 ~ 2026-02).
Signals with longer histories are truncated to this window. Signals starting
after BTC (e.g. defi_tvl_sui from 2023-05) contribute only partial data,
with leading values filled as zero.

### GP Evolution Results (2026-02-26, 448 signals)

Full pipeline run (`evolve --asset BTC --pop-size 500 --generations 30`):

- **5,934 unique expressions** generated in 79s
- **MAP-Elites archive**: 300/10,000 cells (3.0% coverage)
- **Best Sharpe: 1.49** (with transaction cost model)
- **Top 20 all use `if_gt`** — conditional templates provide strong seeds
- Dominant pattern: DeFi regime switching (TUSD supply vs Avalanche TVL)
  as condition, with alternative signals in then/else branches

Key insight: the diversity of 448 signals enables the GP to discover
cross-domain regime-switching alphas that would be impossible with a
handful of traditional signals.

## Live Trading Roadmap

### Goal

Go from paper-only to real BTC trading on Binance with minimal capital,
then scale up as the system proves itself. Target: first real trade
within 2 weeks.

### Current State (2026-02-26)

| Capability           | Alpha-OS                        | Trading-Agent                  |
| -------------------- | ------------------------------- | ------------------------------ |
| Alpha generation     | Complete (DSL + GP + templates) | —                              |
| Walk-Forward CV      | Complete (purged 5-fold)        | —                              |
| Paper trading        | Complete (PaperExecutor + SQLite)| Complete                      |
| Pre-trade risk       | Complete (DD staging + vol target)| —                            |
| Binance connection   | None (AlpacaStub only)          | Complete (CCXT, testnet)       |
| Order execution      | None                            | Complete (book depth, slippage) |
| Circuit breaker      | None                            | Complete (daily loss, DD, kill) |
| Position monitor     | None                            | Complete (SL/TP daemon)        |
| Reconciliation       | None                            | Basic (internal vs exchange)   |

**Conclusion**: Port trading-agent's execution layer into alpha-os.

### Architecture After Integration

```
Alpha-OS (brain)              trading-agent (hands)
┌──────────────────┐         ┌──────────────────┐
│ DSL / GP / Valid. │         │ exchange.py       │ ← CCXT Binance factory
│ Lifecycle / Combo │         │ order_engine.py   │ ← slippage guard
│ RiskManager       │         │ circuit_breaker   │ ← daily loss / DD / kill
│ PaperTrader       │         │ monitor.py        │ ← SL/TP daemon
│                   │         │ reconciliation    │ ← internal vs exchange
│ Executor (ABC)    │         └──────────────────┘
│  ├─ Paper     ✅  │                ↑
│  ├─ Alpaca    stub│                │
│  └─ Binance   NEW ├────────────────┘
└──────────────────┘
```

`BinanceExecutor` wraps trading-agent's CCXT factory + LiveOrderEngine,
conforming to alpha-os's `Executor` interface. The rest of alpha-os
(lifecycle, risk, combination) is unchanged.

### Phases

#### Phase 1: BinanceExecutor (1-2 days) ✅

Create `src/alpha_os/execution/binance.py`:

- Import trading-agent's `exchange.create_spot_exchange()` (CCXT factory)
- Adapt `LiveOrderEngine` logic to alpha-os `Executor.submit_order()` interface
- Credentials from `~/.secrets/binance` (env override supported)
- Testnet mode by default (`testnet=True`)

```
Executor (ABC)
├── PaperExecutor    ← existing, unchanged
├── AlpacaExecutor   ← existing stub, untouched
└── BinanceExecutor  ← new: CCXT spot + slippage guard
```

#### Phase 2: Circuit Breaker (1 day) ✅

Create `src/alpha_os/risk/circuit_breaker.py`:

- Port trading-agent's `CircuitBreaker` (daily loss, consecutive losses, max DD)
- Add kill switch file (`data/KILL_SWITCH`) for emergency stop
- Integrate into `RiskManager` as a pre-trade check:
  1. Circuit breaker → halt if tripped
  2. DD staging → scale position (75% / 50% / 25%)
  3. Vol targeting → scale to 15% annualized

State persisted to JSON; survives process restart.

#### Phase 3: CLI `live` Command (1 day) ✅

Based on existing `paper` command:

```bash
# Testnet (default — no real money)
python3 -m alpha_os live --once --testnet

# Production (real money, explicit flag required)
python3 -m alpha_os live --once --real --capital 1000

# Scheduled daily execution
python3 -m alpha_os live --schedule --testnet
```

Reuses PaperTrader's full cycle (data sync → alpha eval → combination →
risk adjustment → execution) with BinanceExecutor instead of PaperExecutor.

#### Phase 4: Testnet Validation (1-2 weeks)

Run `live --testnet --schedule` daily on Binance testnet:

- Verify order fills, slippage measurement
- Trigger circuit breaker intentionally (confirm halt + recovery)
- Run reconciliation (internal positions vs exchange balance)
- Monitor: daily P&L, slippage distribution, fill latency

Success criteria: 10 consecutive days without errors or unexpected state.

#### Phase 5: Production (incremental)

| Week | Capital | Gate                                     |
| ---- | ------- | ---------------------------------------- |
| 1-2  | $500    | System check. Sharpe > 0 to continue     |
| 3-4  | $2,000  | Slippage < 20bps, no circuit breaker trip |
| 5-8  | $5,000  | DD < 10%, stable daily operation          |
| 9+   | Manual  | Scale at discretion                       |

### Design Decisions

- **Binance spot only** (no futures/leverage) — simpler, lower risk, sufficient for daily rebalance
- **Daily rebalance** — no intra-day trading. Matches alpha evaluation frequency
- **Testnet mandatory** — Phase 4 must complete before any real capital
- **Kill switch file** — `touch data/KILL_SWITCH` halts all trading instantly, even if process is unattended
- **No Alpaca** — crypto first; stock trading is a future consideration
- **No dashboard** — CLI logs + SQLite tracker are sufficient for initial operation

### What Is NOT Needed (deferred)

- High-frequency trading — daily rebalance is sufficient
- Web dashboard — CLI logging and `paper --summary` work fine
- Automatic capital scaling — manual decision at each gate
- Multi-exchange — Binance only until system is proven
- Futures/margin — spot only, no leverage

## Multi-Asset Expansion Roadmap

### Vision

Alpha-OS's core engine (DSL → GP evolution → validation → lifecycle) is
asset-agnostic. The same pipeline that discovers BTC alphas can discover
edges in ETH, SOL, US equities, and beyond. The goal is to expand across
uncorrelated markets to diversify returns while reusing the same
infrastructure.

### Asset Universe

Already defined in `data/universe.py`:

| Class         | Assets                                   | Data Source     | Executor     |
| ------------- | ---------------------------------------- | --------------- | ------------ |
| Crypto spot   | BTC, ETH, SOL, BNB, XRP, ADA, DOGE       | signal-noise    | Binance      |
| US equities   | NVDA, AAPL, MSFT, GOOGL, AMZN, META, TSLA, AMD | signal-noise | Alpaca  |
| Macro signals | VIX, DXY, gold, oil, S&P500, Nasdaq, bonds | signal-noise  | (input only) |
| Alternative   | earthquake, sunspot, ENSO, mempool, hashrate | signal-noise | (input only) |

Macro and alternative signals are used as **inputs to alpha expressions**,
not as tradable assets. They improve alpha quality by enabling cross-domain
regime detection (e.g., `(if_gt vix_close 30.0 (neg btc_ohlcv) (roc_10 btc_ohlcv))`).

### Phase 6: Multi-Crypto (parallel BTC operation)

Run independent alpha-evolution + trading cycles for each crypto asset.
Each asset gets its own registry, lifecycle, and risk budget.

```
cron: 0 */4 * * *
  ├── run_live.sh --asset BTC
  ├── run_live.sh --asset ETH
  └── run_live.sh --asset SOL
```

Implementation:
- Add per-asset DB isolation (`alpha_registry_ETH.db`, etc.) or
  asset column in existing tables
- Extend `run_live.sh` to accept `--asset` parameter
- Independent testnet validation per asset (10 days each)
- Capital allocation: equal-weight initially, then risk-parity

Prerequisite: Phase 4 complete for BTC.

### Phase 7: US Equities via Alpaca

Add `AlpacaExecutor` conforming to `Executor` ABC.

| Feature           | Binance (crypto)          | Alpaca (equities)            |
| ----------------- | ------------------------- | ---------------------------- |
| Market hours      | 24/7                      | 9:30-16:00 ET (Mon-Fri)     |
| Settlement        | Instant                   | T+1                         |
| Commission        | ~0.1%                     | $0                          |
| Min order         | Varies by pair            | Fractional shares supported |
| API               | CCXT                      | alpaca-py SDK               |

Implementation:
- `src/alpha_os/execution/alpaca.py` — replace existing stub
- Market-hours-aware scheduling (skip weekends/holidays)
- Separate credentials: `~/.secrets/alpaca`, `~/.secrets/alpaca_real`
- Paper trading via Alpaca paper endpoint first

Signals already available: NVDA, AAPL, MSFT, GOOGL, AMZN, META, TSLA, AMD
from signal-noise collectors.

### Phase 8: Crypto Futures (leverage)

Add `BinanceFuturesExecutor` for USDT-M perpetual contracts.

Benefits:
- Long **and** short positions (current spot is long-only)
- Leverage (2-5x conservative) amplifies alpha edge
- Funding rate as additional signal input

Risks:
- Liquidation risk — requires tighter circuit breaker thresholds
- Higher complexity in position management

Implementation:
- CCXT supports Binance Futures (`defaultType: "future"`)
- Extend `CircuitBreaker` with liquidation-distance check
- Add funding rate to signal-noise collectors
- New risk parameter: `max_leverage` in config

Prerequisite: Phase 5 profitable for ≥4 weeks on spot.

### Phase 9: Cross-Asset Strategy

Use correlations between asset classes as alpha inputs:

```
# Example: BTC alpha using equity and macro signals
(if_gt (corr_20 sp500 btc_ohlcv) 0.7
  (roc_10 sp500)           # high correlation → follow equities
  (neg (roc_10 dxy)))      # low correlation → inverse dollar
```

Implementation:
- Portfolio-level capital allocator across strategies
- Risk-parity weighting: allocate inversely proportional to volatility
- Cross-asset circuit breaker: halt all if total portfolio DD > threshold
- Unified daily report across all strategies

### Architecture After Expansion

```
alpha-os
├── core/           DSL, GP, backtest, validation (shared)
├── strategy/
│   ├── crypto-spot/     BTC, ETH, SOL (Binance)
│   ├── us-equities/     NVDA, AAPL, ... (Alpaca)
│   └── crypto-futures/  BTC-PERP (Binance Futures)
├── execution/
│   ├── binance.py       Spot executor
│   ├── binance_futures.py  Futures executor
│   └── alpaca.py        Equities executor
├── risk/
│   ├── circuit_breaker.py  Per-strategy + global
│   └── allocator.py        Cross-strategy capital allocation
└── signal-noise         (external) data for all strategies
```

### Priority Order

| Phase | What                 | Why                                     | Prerequisite        |
| ----- | -------------------- | --------------------------------------- | ------------------- |
| 6     | Multi-crypto         | Lowest effort, same infra, diversify    | Phase 4 complete    |
| 7     | US equities          | Large market, zero commission, data ready | Phase 6 stable   |
| 8     | Crypto futures       | Short selling, leverage, higher returns | Phase 5 profitable  |
| 9     | Cross-asset          | Portfolio-level optimization            | Phase 7 stable      |

### What Remains Out of Scope

- **DeFi / on-chain trading** — smart contract risk, MEV, different execution model
- **Forex** — requires separate broker, 24/5 schedule, different market microstructure
- **Options** — pricing model complexity beyond GP's current expressiveness
- **HFT** — daily rebalance is the design choice; sub-minute trading needs different architecture
