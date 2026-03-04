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
| PROBATION → ACTIVE     | Rolling Sharpe ≥ 0.05        |
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
| OOS Sharpe    | ≥ 0.05    | Purged Walk-Forward CV (5 folds)  |
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

## Position Sizing

### Signal Flow

```
alpha_i signals → weighted mean, std → consensus
                                        ↓
direction  = sign(signal_mean)
consensus  = |signal_mean| / (|signal_mean| + signal_std)
adjusted   = direction × consensus × dd_scale
position   = clip(adjusted) × max_position_pct × portfolio_value
```

### Key Components

- **signal_consensus** (`combiner.py`): Measures agreement among alpha
  signals. `consensus = |mean| / (|mean| + std)` — ranges from 1.0 (all
  alphas agree) to 0.0 (equal split between long and short).
- **dd_scale** (`risk/manager.py`): Drawdown-based scaling. Reduces
  position size as portfolio drawdown deepens (3 stages: 5%/10%/15%).

### Design Rationale

1. **No Kelly criterion**: Kelly (μ/σ²) requires stable, repeatable
   returns from the same strategy. This system's alphas are short-lived
   (evolved/replaced every few days), so past portfolio returns reflect
   alphas that no longer exist. Additionally, 4h-cycle returns are too
   small (±0.02%) for μ/σ² to produce meaningful values — it either
   explodes or gets clipped to a constant.
2. **No CVaR gate**: CVaR examines past return distributions, which
   share the same problem as Kelly — past returns come from alphas that
   no longer exist. dd_scale already provides drawdown protection using
   the actual portfolio equity (a fact, not a statistical estimate).
3. **Consensus modulates risk**: When alphas disagree, consensus → 0 and
   position shrinks automatically. This uses only "right now" information,
   so alpha turnover is irrelevant.
4. **dd_scale stacks with consensus**: Drawdowns reduce sizing directly.
   Both are real-time signals that don't depend on historical strategy
   stability.

### Known Limitation: Top-30 Instability

The current system generates ~144,000 alpha candidates/day (500/round ×
5min intervals). The top 30 trading alphas change almost daily — average
alpha lifetime is ~1.6 days (ACTIVE), ~3.4 days (PROBATION). This means
signal_consensus measures agreement among a different set of alphas each
cycle, reducing its reliability as a sizing signal.

Two independent approaches are being considered:

- **Path A (stabilize)**: Reduce generation rate, raise adoption bar,
  add tenure bonus to lifecycle scoring — make the top 30 stable so
  consensus becomes meaningful. This improves the current alpha-os.
- **Path B (turnover-native)**: Build a separate system designed for
  high-turnover alpha statistical voting. Instead of relying on a stable
  top 30, treat all alphas as ephemeral voters. This would be a new
  project, not a layer on top of alpha-os.

These are independent: Path A improves alpha-os as-is, Path B is a
separate system for a different paradigm. They are not layered.

## Paper Trading

Two modes:

- **Daily cycle** (`paper --once`): Syncs data, evaluates all alphas,
  applies lifecycle transitions, combines signals with quality × diversity
  weights, adjusts for risk (drawdown + volatility scaling), and executes
  via paper executor.
- **Backfill simulation** (`paper --backfill`): Vectorized historical
  simulation. Pre-computes all signal matrices once, then iterates days
  via array indexing. Orders of magnitude faster than per-day evaluation.

Position sizing uses signal consensus × drawdown scaling (see Position
Sizing section above).

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
- **Crypto first** — BTC spot as initial proving ground; equities, commodities, prediction markets to follow
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

### Future Asset Classes (not yet implemented)

The core engine (DSL → GP → validation → lifecycle) is asset-agnostic.
Each new asset class requires: (1) signal-noise collectors, (2) an Executor implementation, (3) asset-specific risk parameters.

#### Tradable — Near-term

| Asset Class | Examples | Executor | Data Readiness |
|-------------|----------|----------|---------------|
| REITs / ETFs | VNQ, TLT, GLD, SPY | Alpaca / IBKR | signal-noise に一部あり |
| Bonds / Fixed Income | 米国債ETF (SHY, IEF, TLT), 社債ETF | Alpaca / IBKR | 利回りシグナル既存 |
| Commodities (broad) | 金, 銀, プラチナ, 銅, 原油, 天然ガス | 先物 or ETF (GLD, SLV, USO) | 金・原油シグナル既存 |
| Agricultural | 小麦, トウモロコシ, 大豆, コーヒー | 先物 or ETF (WEAT, CORN) | 要 collector 追加 |
| Forex | USD/JPY, EUR/USD | OANDA / IBKR | 要 collector 追加 |

#### Tradable — Medium-term

| Asset Class | Examples | Executor | Notes |
|-------------|----------|----------|-------|
| Prediction markets | Polymarket, Kalshi | Custom API | バイナリーアウトカム; 独自の DSL 拡張が必要になる可能性 |
| Options / Derivatives | BTC options (Deribit), 株式オプション | Deribit / IBKR | IV surface, Greeks; Phase 4 で IV シグナルは計画済み |
| Crypto futures | BTC-PERP, ETH-PERP | Binance Futures | ショート可、レバレッジ; Phase 8 で計画済み |
| Volatility products | VIX先物, UVXY, SVXY | IBKR / 先物 | VIX シグナル既存; トレード対象としては未 |

#### Exploratory — Long-term

| Asset Class | Examples | Notes |
|-------------|----------|-------|
| DeFi / on-chain | DEX swap, lending yield | スマートコントラクトリスク, MEV |
| Carbon credits | EU ETS, CCA | 新興市場, 流動性限定的 |
| Sports betting | ブックメーカー API | Polymarket と同構造 |
| Tokenized RWA | 不動産・債権トークン | 規制・流動性の課題 |
| Energy markets | 電力先物, ウラン | 専門的な市場構造 |

#### Design Constraint

- **HFT** — daily rebalance is the design choice; sub-minute trading needs different architecture

## Pipeline Architecture v2: Separated Processes

### Motivation

The current pipeline is a batch monolith: every 4 hours, a single process
runs evolve → validate → adopt → combine → risk → trade sequentially.
This design emerged from the initial prototype and served well through
Phase 1-3, but operational experience on the alpha-os server (cx23, then
cx33) has revealed fundamental limitations:

1. **Memory spikes**: GP evolution generates 10,000 candidates in a single
   burst, causing memory to spike from ~300MB to 1GB+. On the original
   cx23 (4GB), this consumed 800MB of swap while signal-noise was co-located.

2. **Exploration blocks execution**: The evo phase consumes CPU 87% and
   must complete before any trading can occur. A trade cycle that should
   take seconds is blocked for minutes by evolution.

3. **God class**: `Trader.run_cycle()` is 250+ lines spanning alpha
   evaluation, lifecycle transitions, diversity recomputation, signal
   combination, risk adjustment, and order execution. Testing and
   modification require understanding the entire flow.

4. **O(n²) diversity**: With 22,625 active alphas, computing the full
   correlation matrix for diversity scoring requires ~256M pair
   comparisons. This will not scale to 100K+ alphas.

5. **Wasteful data reload**: Each cycle loads the full signal matrix
   (2,089 days × 448 features ≈ 7.5MB) even though only one new row
   was added since the last cycle.

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                    SQLite (WAL mode, shared state)                    │
│  ┌──────────────┬────────────────┬──────────────┬─────────────────┐  │
│  │ alpha_cache   │ alpha_registry │ forward_rets │  paper_trading  │  │
│  │ (signals)     │ (alphas)       │ (tracking)   │  (snapshots)    │  │
│  │               │ candidates NEW │              │                 │  │
│  │               │ div_cache  NEW │              │                 │  │
│  └──────────────┴────────────────┴──────────────┴─────────────────┘  │
└──────▲───────────────▲──────────────────▲────────────────▲───────────┘
       │               │                  │                │
┌──────┴──────┐  ┌─────┴──────┐  ┌───────┴───────┐  ┌─────┴──────────┐
│ evo daemon  │  │  validator │  │  trade cycle  │  │   lifecycle    │
│ (continuous)│  │ (triggered)│  │   (4h 周期)   │  │   (daily)      │
│             │  │            │  │               │  │                │
│ GP evolve   │  │ purged CV  │  │ read alphas   │  │ forward rets   │
│ pop=80      │  │ DSR, PBO   │  │ read div_cache│  │ rolling Sharpe │
│ 15 gens     │  │ gate       │  │ combine       │  │ state trans.   │
│ → candidates│  │ diversity  │  │ risk adjust   │  │ → alphas.state │
│   table     │  │ → alphas   │  │ execute       │  │                │
│             │  │ → div_cache│  │               │  │                │
└─────────────┘  └────────────┘  └───────────────┘  └────────────────┘
       │                                   │
       │         ┌──────────────────┐      │
       └────────→│  signal-noise    │←─────┘
         read    │ REST 127.0.0.1   │  sync
         only    │ :8000 (既存)     │
                 └──────────────────┘
```

Data flow:

1. **evo daemon** runs GP evolution continuously in small batches and writes
   candidates to the `candidates` table.
2. **validator** reads PENDING candidates, runs statistical validation
   (purged WF-CV, DSR, PBO, adoption gate), computes incremental diversity
   against existing ACTIVE alphas, and writes to `alphas` (ACTIVE) and
   `diversity_cache`.
3. **trade cycle** reads ACTIVE/PROBATION alphas and diversity cache,
   computes weighted combination, applies risk adjustments, and executes
   via Binance. No evolution, no lifecycle evaluation.
4. **lifecycle manager** reads forward returns, computes rolling Sharpe
   (63-day window), and transitions alpha states (ACTIVE → PROBATION →
   DORMANT or revival).

### Process Definitions

#### evo daemon (continuous)

Responsibility: continuously explore the alpha search space and feed
candidates to the validator.

Extracts logic from `PipelineRunner._evolve()` (pipeline/runner.py) and
`GPEvolver.run()` (evolution/gp.py) into a long-lived daemon.

| Parameter | Current (batch) | New (daemon) |
|-----------|----------------|--------------|
| pop_size | 200 | 80 |
| n_generations | 30 | 15 |
| Execution | 1 burst per 24h | Continuous rounds |
| Memory pattern | Spike to 1GB+ | Flat ~200-400MB |
| Archive | Discarded per run | In-process, reset on restart |

Implementation: `src/alpha_os/daemon/evo.py`

- Each round: initialize population → evolve 15 generations → collect
  results → bulk insert into `candidates` table → gc.collect() → sleep.
- MAP-Elites archive is maintained in process memory across rounds for
  quality-diversity coverage. Reset on process restart is acceptable
  because all candidates are persisted in the `candidates` table.
- Memory guard: check `resource.getrusage(RUSAGE_SELF).ru_maxrss` after
  each generation. If RSS exceeds `memory_limit_mb`, halve pop_size and
  prune the archive.
- Graceful shutdown on SIGTERM (reuse existing `PipelineScheduler`
  signal handling pattern).

CLI entry point:

```
python -m alpha_os evo-daemon --asset BTC [--config ...]
```

#### validator (triggered batch)

Responsibility: statistically validate candidates and register surviving
alphas.

Extracts logic from `PipelineRunner._validate()` and `_adopt()`
(pipeline/runner.py).

- Polls `candidates` table every `poll_interval` seconds (default: 1800).
- When PENDING count ≥ `min_queue_size` (default: 10), fetches a batch
  (default: 100 candidates).
- For each candidate:
  1. Parse expression, evaluate on current data.
  2. Purged Walk-Forward CV (5 folds, embargo 5 days).
  3. Deflated Sharpe Ratio.
  4. Batch PBO (computed once per batch).
  5. Adoption gate (OOS Sharpe, PBO, DSR, correlation, min_days).
- Candidates that pass: `registry.register()` as ACTIVE, update
  `candidates.status = 'adopted'`.
- Candidates that fail: update `candidates.status = 'rejected'`.

**Incremental diversity computation** (key optimization):

Instead of computing the full N×N correlation matrix, the validator
computes correlations only between new candidates and existing ACTIVE
alphas. For a batch of 100 new candidates against 50,000 ACTIVE alphas,
this is 5M pairs instead of 2.5 billion — a 500× reduction.

Results are written to the `diversity_cache` table. The trade cycle reads
this cache directly instead of recomputing diversity.

Implementation: `src/alpha_os/daemon/validator.py`

CLI entry point:

```
python -m alpha_os validator --asset BTC [--config ...]
```

#### trade cycle (periodic 4h) — existing Trader, slimmed down

Responsibility: combine active alphas, apply risk adjustments, execute.

This is the existing `Trader.run_cycle()` (paper/trader.py) with two
responsibilities removed:

1. **Lifecycle evaluation** (monitor.check → lifecycle.evaluate → state
   transitions): moved to lifecycle manager.
2. **Diversity recomputation** (_recompute_diversity): moved to validator.

What remains in run_cycle():

```
Pre-checks: cooldown, circuit breaker
Phase 1:    store.sync() → get_matrix() → evaluate ACTIVE alphas
            → forward_tracker.record()
Phase 2:    read diversity_cache table → compute_weights()
            → weighted_combine_scalar()
Phase 3:    risk adjustments (consensus, dd_scale, regime)
Phase 4:    executor.rebalance() → portfolio_tracker.save_snapshot()
```

Backward compatibility: `skip_lifecycle` parameter (default: False).
When `evo_daemon.enabled = true` in config, the trade cycle sets
`skip_lifecycle = True` automatically.

No new CLI command — uses existing `live --schedule`.

#### lifecycle manager (daily)

Responsibility: evaluate forward performance of all alphas and transition
states.

Extracts the lifecycle evaluation loop from `Trader.run_cycle()` lines
304-336 and the equivalent logic in `ForwardRunner.run_cycle()`.

- Runs daily at UTC 00:30 via systemd timer.
- Reads all ACTIVE, PROBATION, and DORMANT alphas from registry.
- For each alpha: reads forward returns (63-day window) from
  `forward_returns` table → computes rolling Sharpe via
  `AlphaMonitor.check()` → calls `AlphaLifecycle.evaluate()` → updates
  `alphas.state` if transition occurs.
- Logs state changes to audit log.

Memory: ~14MB for 22,625 alphas × 63 days × 8 bytes. Completes in
seconds.

Implementation: `src/alpha_os/daemon/lifecycle.py`

CLI entry point:

```
python -m alpha_os lifecycle --asset BTC [--config ...]
```

### Communication: SQLite as Shared State

No external message queue (Redis, RabbitMQ) is needed. SQLite with WAL
mode provides sufficient concurrent access for 4 processes.

#### New table: `candidates`

Added to `alpha_registry.db`:

```sql
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    expression TEXT NOT NULL,
    fitness REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    -- status: 'pending' | 'validating' | 'adopted' | 'rejected'
    oos_sharpe REAL,
    pbo REAL,
    dsr_pvalue REAL,
    behavior_json TEXT DEFAULT '{}',
    created_at REAL NOT NULL,
    validated_at REAL,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_candidates_status
    ON candidates(status);
CREATE INDEX IF NOT EXISTS idx_candidates_created
    ON candidates(created_at);
```

Garbage collection: delete `adopted` and `rejected` rows older than 30
days. Adopted candidates are already in the `alphas` table.

#### New table: `diversity_cache`

Added to `alpha_registry.db`:

```sql
CREATE TABLE IF NOT EXISTS diversity_cache (
    alpha_id TEXT PRIMARY KEY,
    diversity_score REAL NOT NULL,
    computed_at REAL NOT NULL,
    n_alphas_compared INTEGER NOT NULL
);
```

Updated by validator when new alphas are adopted. Read by trade cycle
for weight computation.

#### WAL mode for all databases

Currently only `DataStore` (alpha_cache.db) enables WAL mode. Extend to
all databases for multi-process safety:

| Database | WAL | busy_timeout | Writer(s) | Reader(s) |
|----------|-----|-------------|-----------|-----------|
| alpha_cache.db | ✅ existing | 30s | trade cycle (sync) | evo daemon |
| alpha_registry.db | **NEW** | 30s | evo (candidates), validator (alphas, div_cache), lifecycle (alphas.state) | trade cycle |
| forward_returns.db | **NEW** | 30s | trade cycle (record) | lifecycle |
| paper_trading.db | **NEW** | 30s | trade cycle | — |

Write contention is limited to `alpha_registry.db` where validator and
lifecycle may both update `alphas`. With `busy_timeout=30000`, SQLite
retries for up to 30 seconds, which is more than sufficient for the
millisecond-scale writes involved.

### Configuration

New TOML sections in `config/default.toml`:

```toml
[evo_daemon]
enabled = false            # true activates separated process mode
pop_size = 80              # GP population per round
n_generations = 15         # generations per round
round_interval = 300       # seconds between rounds (5 min)
memory_limit_mb = 400      # soft RSS limit, halve pop if exceeded
batch_size = 500           # max candidates per round

[validator]
enabled = false
poll_interval = 1800       # seconds between queue checks (30 min)
batch_size = 100           # candidates per validation batch
min_queue_size = 10        # minimum queue depth to trigger
diversity_recompute_days = 63   # full recalculation interval
incremental_diversity = true    # new-vs-existing only

[lifecycle_daemon]
enabled = false
# When enabled, Trader.run_cycle() skips inline lifecycle evaluation.
# Lifecycle transitions are handled by the lifecycle daemon instead.
```

All sections default to `enabled = false`, preserving the existing
monolithic behavior. The separated process architecture is activated by
setting `enabled = true` in the relevant sections.

### Incremental Migration Path

Phase 4 (testnet validation) is in progress. The migration must minimize
trade interruption.

#### Step 1: Foundation (no trade downtime)

- Add WAL mode + `busy_timeout` to `AlphaRegistry`, `ForwardTracker`,
  `PaperPortfolioTracker` (3 files, ~6 lines each).
- Add `candidates` and `diversity_cache` tables to
  `AlphaRegistry._create_tables()`.
- Create `daemon/` package with `__init__.py`.
- Add `[evo_daemon]`, `[validator]`, `[lifecycle_daemon]` to TOML and
  corresponding dataclasses to `config.py`.
- Run all 434 existing tests to confirm no regression.

#### Step 2: evo daemon (no trade downtime)

- Implement `daemon/evo.py`. Extract logic from `PipelineRunner._evolve()`.
- Add `evo-daemon` CLI subcommand.
- Add `alpha-os-evo@.service` systemd unit.
- Test: start evo daemon, verify candidates appear in DB.
- The existing `alpha-os.service` continues running unchanged.

#### Step 3: validator (no trade downtime)

- Implement `daemon/validator.py`. Extract logic from
  `PipelineRunner._validate()` and `_adopt()`.
- Add `validator` CLI subcommand.
- Add `alpha-os-validator@.service` systemd unit.
- Test: verify candidates flow through validation to alphas table.

#### Step 4: trade cycle slimming (~1 min downtime)

- Add `skip_lifecycle` parameter to `Trader.run_cycle()`.
- Switch diversity source to `diversity_cache` table when
  `validator.enabled = true`.
- Condition evolution in `cli.py:cycle()` on `evo_daemon.enabled`.

```python
# cli.py: cycle()
if evolve_interval > 0 and not cfg.evo_daemon.enabled:
    # Legacy: inline evolution
    ...
```

- Set `enabled = true` in TOML, restart `alpha-os.service`. Downtime: ~1
  minute.
- Rollback: set `enabled = false`, restart. Instant revert.

#### Step 5: lifecycle manager (no trade downtime)

- Implement `daemon/lifecycle.py`.
- Add `lifecycle` CLI subcommand.
- Add `alpha-os-lifecycle@.service` + `.timer`.
- Set `lifecycle_daemon.enabled = true`. Trade cycle stops doing inline
  lifecycle evaluation.

#### Step 6: legacy cleanup (after Phase 4 complete)

- Remove inline evolution path from `cli.py:cycle()`.
- Remove lifecycle code from `Trader.run_cycle()`.
- Remove `skip_lifecycle` flag (always skip).
- Simplify `PipelineRunner.run()` to use evo + validate internally (keep
  for testing convenience).

### Scaling to 100K+ Alphas

#### Diversity: O(n²) → O(new × existing)

Current `compute_diversity_scores()` computes `(N, T)` correlation
matrix for all alpha pairs. At N=100K, this is 5 billion pairs — infeasible.

The validator's incremental approach computes correlation only between new
candidates (batch of ~100) and existing ACTIVE alphas (N). For N=100K,
this is 10M pairs per batch — 500× cheaper.

For periodic full recalculation (every 63 days), use random sampling:
select 5,000 representative alphas from the ACTIVE pool and compute
diversity against this sample. Approximation error is bounded and
acceptable for portfolio weighting.

#### Alpha evaluation: skip low-weight alphas

In trade cycle, alphas with weight equal to `min_weight` (1e-4) contribute
negligibly to the combined signal. With 100K alphas, most will be at
min_weight. Skip evaluation of alphas whose cached weight is below a
threshold (e.g., 10× min_weight = 1e-3).

This reduces evaluation from O(N) to O(N_effective), where N_effective is
the number of alphas with meaningful weight (typically ~1,000-5,000).

#### Data loading: windowed get_matrix()

Add `start` parameter to `DataStore.get_matrix()`:

```python
matrix = store.get_matrix(features, start=today - 252, end=today)
```

The trade cycle needs at most `corr_lookback=252` days for volatility
scaling and monitor checks. Loading 252 days instead of 2,089 reduces
memory by 8×.

For evo daemon, the full history may be needed for backtesting. This is
acceptable because evo runs in a separate process with its own memory
budget.

### systemd Service Design

#### Service definitions

**alpha-os-evo@.service** (continuous):

```ini
[Unit]
Description=Alpha-OS evolution daemon (%i)
After=network.target signal-noise-scheduler.service

[Service]
Type=simple
User=dev
WorkingDirectory=/home/dev/projects/alpha-os
ExecStart=/home/dev/projects/alpha-os/.venv/bin/python \
    -m alpha_os evo-daemon --asset %i
Restart=on-failure
RestartSec=60
MemoryHigh=400M
MemoryMax=600M
StandardOutput=append:/home/dev/projects/alpha-os/data/%i/logs/evo.log
StandardError=append:/home/dev/projects/alpha-os/data/%i/logs/evo.log
KillSignal=SIGTERM
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
```

**alpha-os-validator@.service** (continuous poller):

```ini
[Unit]
Description=Alpha-OS validator daemon (%i)
After=network.target signal-noise-scheduler.service

[Service]
Type=simple
User=dev
WorkingDirectory=/home/dev/projects/alpha-os
ExecStart=/home/dev/projects/alpha-os/.venv/bin/python \
    -m alpha_os validator --asset %i
Restart=on-failure
RestartSec=120
MemoryHigh=500M
MemoryMax=700M
StandardOutput=append:/home/dev/projects/alpha-os/data/%i/logs/validator.log
StandardError=append:/home/dev/projects/alpha-os/data/%i/logs/validator.log
KillSignal=SIGTERM
TimeoutStopSec=300

[Install]
WantedBy=multi-user.target
```

**alpha-os-lifecycle@.service** + **alpha-os-lifecycle@.timer** (daily):

```ini
# alpha-os-lifecycle@.service
[Unit]
Description=Alpha-OS lifecycle evaluation (%i)

[Service]
Type=oneshot
User=dev
WorkingDirectory=/home/dev/projects/alpha-os
ExecStart=/home/dev/projects/alpha-os/.venv/bin/python \
    -m alpha_os lifecycle --asset %i
MemoryHigh=300M
MemoryMax=500M
StandardOutput=append:/home/dev/projects/alpha-os/data/%i/logs/lifecycle.log
StandardError=append:/home/dev/projects/alpha-os/data/%i/logs/lifecycle.log

# alpha-os-lifecycle@.timer
[Unit]
Description=Daily lifecycle evaluation for %i

[Timer]
OnCalendar=*-*-* 00:30:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
```

#### Memory budget (cx33: 8GB RAM)

| Service | MemoryHigh | MemoryMax | Notes |
|---------|-----------|-----------|-------|
| signal-noise scheduler | ~700M | — | Existing, unchanged |
| signal-noise serve | ~300M | — | Existing, unchanged |
| alpha-os trade cycle | 300M | 500M | Slimmed (was ~500M with inline evo) |
| alpha-os-evo | 400M | 600M | Memory spike isolated here |
| alpha-os-validator | 500M | 700M | Validation + diversity computation |
| alpha-os-lifecycle | 300M | 500M | Daily oneshot, short-lived |
| OS + system services | ~500M | — | systemd, tailscale, sshd, etc. |
| **Total MemoryHigh** | **~3000M** | | Lifecycle and validator rarely overlap |

Effective peak usage is ~2.5GB. 8GB provides comfortable headroom with
no swap usage expected.

### Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| SQLite BUSY on alpha_registry.db | Validator and lifecycle write to `alphas` concurrently | WAL mode + `busy_timeout=30000ms`. Writes are millisecond-scale; 30s retry is more than sufficient. |
| evo daemon memory leak | OOM kill → candidates lost | systemd `MemoryMax=600M` hard cap. RSS monitoring + pop_size halving in daemon. `Restart=on-failure` for auto-recovery. Candidates already written to DB survive. |
| Phase 4 testnet disruption | Testnet validation streak resets | Steps 1-3 add no risk (new code only). Step 4 uses `enabled=false` default. Rollback: flip flag + restart (~1 min). |
| candidates table bloat | DB file growth | GC job deletes `adopted`/`rejected` rows older than 30 days. Scheduled inside validator daemon. |
| diversity cache staleness | Trade weights based on outdated diversity | `computed_at` timestamp tracked. Cache valid for `diversity_recompute_days` (63 days). DORMANT alphas excluded from combination regardless of cache state. |
| Process ordering | Trade runs before validator finishes first batch | Not a problem. Trade uses whatever ACTIVE alphas exist at cycle time. New alphas appear in the next cycle automatically. Exploration and exploitation are intentionally decoupled. |

### Event-Driven Mode

The existing `EventDrivenTrader` (paper/event_driven.py) subscribes to
signal-noise WebSocket events and triggers trade cycles on market events
instead of a fixed timer. This mode is orthogonal to the process
separation described above — it replaces the trade cycle's timer
trigger, not its internal logic.

Integration with v2 architecture is deferred. The event-driven trigger
mechanism can be applied to the slimmed trade cycle in a future iteration.

### Future Considerations

- **PostgreSQL migration**: if alpha count exceeds 500K or multi-machine
  deployment is needed, migrate from SQLite to PostgreSQL for true
  concurrent writes.
- **Multiprocessing within evo**: parallelize GP evaluation using
  `multiprocessing.Pool` within the evo daemon for multi-core utilization.
- **Cross-asset portfolio optimization**: a fifth process that reads
  positions across all per-asset trade cycles and applies portfolio-level
  risk-parity allocation.
