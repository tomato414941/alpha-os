# Alpha-OS System Design

This document captures architectural rationale and target design direction.
It is not the source of truth for the current recovery posture or current
runtime entrypoints.

For current operating truth, prefer:

- `README.md` for the current runtime path
- `RECOVERY.md` for trust boundaries, scheduler policy, and recovery order
- `docs/portfolio-runtime-principles.md` for current portfolio / allocation terminology

## Glossary

### Core terms

| Term | Definition | Examples |
|------|-----------|---------|
| **feature** | Input data series from signal-noise. Raw or computed market observables that hypotheses consume. | `fear_greed`, `btc_ohlcv`, `vix_close`, `funding_rate_btc` |
| **hypothesis** | A single predictive logic that consumes features and produces predictions. A claim about market inefficiency; it may or may not have real predictive power. | `(sub fear_greed dxy)`, XGBoost model, RSI mean-reversion rule |
| **prediction** | The concrete output value a hypothesis produces for a given date and asset. This is what the pipeline evaluates. | `+0.3`, `-0.15` |
| **alpha** | Excess return over benchmark. Not a method but a **result** — the return attributable to skill rather than market exposure. A hypothesis may or may not generate alpha; this is discovered through evaluation, not assumed. | Sharpe 0.5 after subtracting benchmark |

**feature → hypothesis → prediction** is the data flow. The pipeline
evaluates predictions, never hypothesis internals.

### Hierarchy

```
strategy (any method of generating profit)
│
│   prediction reliance is a spectrum, not a binary:
│
├── strongly predictive — prediction is the core value
│   └── hypothesis (individual instance)
│       ├── DSL expression
│       ├── ML model
│       ├── classical indicator
│       └── ...
└── weakly predictive — structure or speed is the core value,
    │                    prediction is auxiliary
    ├── arbitrage ("these prices will converge")
    ├── market making ("spread will exceed adverse selection")
    └── ...

Pure non-predictive strategies exist only in textbooks.
Real implementations almost always involve some prediction.

alpha (excess return) is orthogonal — it is the outcome, not the method.
Any strategy can generate alpha. Whether a hypothesis has alpha is
discovered through IC/Sharpe evaluation, not assumed from its form.
```

alpha-os is a system for strongly predictive strategies. It manages
hypotheses, evaluates their predictions, and allocates capital based
on results.

### Multi-strategy extensibility

The current system is a single predictive sleeve. The name "alpha-os"
is goal-oriented (generate alpha), not method-oriented, so it can
accommodate non-predictive strategies in the future.

Different strategy types have fundamentally different internal
pipelines. The only universal output is P&L.

```
alpha-os platform
├── sleeve: predictive (current)
│   feature → hypothesis → prediction → P&L
├── sleeve: arbitrage (future)
│   price feeds → spread detection → P&L
├── sleeve: market making (future)
│   order book → quoting → P&L
└── ...

shared layer (applies to all sleeves):
  capital allocator — cross-sleeve allocation (Sharpe, correlation, drawdown)
  risk manager — portfolio-level limits
  execution — shared order infrastructure
  data — shared market data
```

Each sleeve owns its internal pipeline. The platform evaluates sleeves
by P&L characteristics, not by prediction accuracy. IC evaluates
hypotheses within the predictive sleeve; Sharpe/drawdown evaluates
sleeves at the platform level.

This is the pod model used by multi-strategy hedge funds. The current
alpha-os is one pod. Adding sleeves requires only the shared layer
interfaces (P&L reporting, risk limits, capital requests), not changes
to the predictive pipeline.

### Hypothesis dimensions

A hypothesis is defined by two orthogonal axes: **method** (how it is
built) and **domain** (what market phenomenon it targets). These
combine freely — any method can target any domain.

#### Method (how)

| Method | What it does | Status |
|--------|-------------|--------|
| **DSL / GP** | Compose features via S-expression operators, evolved by genetic programming | Active — primary generator |
| **Human-authored** | Fixed rules written by humans from domain knowledge | Active — classical_producer, seed-handcrafted |
| **ML** | Learn patterns from features statistically | Planned |
| **LLM / NLP** | Extract predictions from unstructured text | Future |
| **Meta / ensemble** | Combine other hypotheses' predictions (stacking) | Future |
| **External ingest** | Import predictions from outside systems | Future |

#### Domain (what)

| Domain | What it targets | Examples |
|--------|----------------|---------|
| **Technical / macro** | Price patterns, macro indicators | RSI, momentum, carry, dollar weakness |
| **Options** | Volatility surface, skew, term structure | IV smile interpretation |
| **Order flow** | Microstructure signals | VPIN, book imbalance, trade flow |
| **On-chain** | Blockchain observables | Wallet flows, whale behavior, DeFi TVL |
| **Event** | Calendar and news events | FOMC, halving, earnings |
| **Lead-lag** | Cross-asset time delays | ETH lagging BTC by N hours |
| **External markets** | Prediction markets, analyst forecasts | Polymarket odds |

#### Examples

| Hypothesis | Method | Domain |
|-----------|--------|--------|
| `(sub fear_greed dxy)` | DSL / GP | Technical / macro |
| RSI mean-reversion rule | Human-authored | Technical / macro |
| XGBoost on order book features | ML | Order flow |
| GPT sentiment from earnings calls | LLM / NLP | Event |
| Polymarket odds as directional signal | External ingest | External markets |

The boundary between feature and domain depends on processing depth.
A single time series (`iv_skew_btc`) is a feature. Logic that
interprets it to produce a directional prediction is a hypothesis.

Note: the `alpha` package is now a legacy compatibility boundary. In
the current terminology, `alpha` is reserved for excess return as an
outcome, while predictive units are hypotheses. `AlphaRecord`,
`alpha_id`, and similar identifiers remain in code for now as legacy
names. New code and documentation should prefer `hypothesis` for the
predictive unit and should not add new source-of-truth logic under
`alpha/`.

## Principles

**1. Prediction-first, metadata-enriched.**
Evaluation is based on predictions alone — "you predicted X, did X
happen?" DSL expression, ML model, LLM, human — all evaluated the
same way. A hypothesis that provides no metadata is still fully
functional in the pipeline.

However, hypotheses may optionally provide **feature dependency
metadata** (which features they consume, importance weights). When
available, the platform uses this for proactive risk management:
detecting feature concentration, alerting before a data source
failure cascades through correlated hypotheses, and monitoring
diversity across the portfolio.

This metadata is cheap to produce (DSL expressions are fully
inspectable, ML models expose feature importances) and valuable
for risk. The pipeline does not use it for evaluation or selection
— only for monitoring and risk.

What this eliminates: complexity penalties, semantic duplicate
detection based on structure. Redundancy is handled by marginal
contribution (identical outputs → zero marginal → natural death),
not by structural inspection.

**2. Adaptive and online.**
Evaluation is continuous, not batch. Every day, every hypothesis's
latest prediction is scored against realized outcomes. No periodic
retraining cycles or manual review gates. The system learns what
works and what doesn't, always, automatically.

**3. IC for hypotheses, Sharpe for portfolio.**
Individual hypothesis quality is measured by IC (prediction accuracy).
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

### 3. Admit (Noise Filter)

Admission is a practical noise filter, not a quality judgment. Generator
produces thousands of candidates per round — most are noise. Admitting
all to the live pool would waste computation.

- Purged walk-forward CV: compute OOS IC per fold at the stored horizon
- IC must be consistently positive across folds
- Statistical checks (PBO, DSR) as sanity filters
- Full eval universe (~919 assets): per-asset IC stored for later use

Role: minimum viable prediction quality. Like a prediction market's
minimum bet — not judging the bettor, just filtering zero-value entries.

Signals that pass enter the live pool with a **minimum stake**.

### 4. Live Selection (Internal Market)

Each live signal carries a virtual stake. Stake grows or shrinks based
on the signal's **marginal contribution to portfolio P&L** — not its
individual accuracy. This replaces TC weighting, lifecycle state machine,
and manual thresholds with a single mechanism.

#### Normalization

Signals are normalized by the pipeline before combination. Generators
output raw values on arbitrary scales; the pipeline applies uniform
normalization (rank or z-score) so signals are combinable. This is
the pipeline's responsibility, not the generator's.

Features are also normalized by the pipeline before being given to
generators. This makes DSL operations like `add` and `sub` meaningful
across features of different scales, and is a practical requirement
for ML generators.

#### Multi-asset portfolio

Each signal participates in multiple assets' portfolios based on
per-asset IC (stored at admission). Stake is per signal, not per
signal-asset pair.

```
every day:
    for each traded asset a:
        # Signals relevant to this asset (per_asset_ic[a] > 0)
        relevant = signals where per_asset_ic[j][a] > 0
        portfolio_a = Σ(stake_j × normalized_signal_j
                        for j in relevant) / Σ(stake_j)
        pnl_a = portfolio_a × realized_residual_return_a

    # Capital allocation across assets (risk parity or IC-weighted)
    total_pnl = Σ(allocation_a × pnl_a)

    for each live signal j:
        marginal_j = total_pnl_with_j - total_pnl_without_j
        record marginal_j in signal j's rolling history

    # Stake = recent performance, not cumulative
    for each live signal j:
        stake_j = mean(marginal history over last N days)  # e.g. N=60
        if stake_j < min_stake: remove signal  # natural death

    cap any single signal at max_weight (e.g. 5%)
```

A signal effective across 50 assets has higher marginal contribution
than one effective for 1 asset. Generalization is naturally rewarded.

Portfolio is a linear combination per asset, so removing signal j
is O(1) per asset. 500 signals × 50 assets = 25,000 subtractions
per day. Trivial.

#### Why marginal contribution, not individual P&L

Individual P&L (`sign(signal) * return`) has a fatal flaw: two
identical signals get the same score and both grow. The portfolio
doubles its bet on one prediction without knowing it.

Marginal contribution fixes this. If signal j is identical to another
signal already in the portfolio, removing j changes nothing.
`marginal_j = 0`. Stake stagnates. Signal dies. Redundancy is
eliminated automatically — no duplicate detection needed.

**This also solves non-return prediction targets.** A volatility
signal doesn't predict direction — it adjusts position sizing. Its
value shows up as improved portfolio P&L (smaller positions before
crashes, larger positions in calm periods). Marginal contribution
captures this: "with the vol signal, the portfolio made more money
than without it."

**Turnover is embedded, not separate.** The portfolio rebalances
based on signals. High-turnover signals cause frequent rebalancing →
transaction costs reduce portfolio P&L → their marginal contribution
shrinks. No explicit turnover metric needed.

**Tail risk protection:**
- Single signal weight capped (e.g. 5% max)
- Large loss → immediate stake reduction → automatic de-risking
- Portfolio-level circuit breaker (separate, see Risk Management)

#### Rolling window, not cumulative

Stake is the mean marginal contribution over a rolling window (e.g.
60 days), not a cumulative compounding value.

**Why:** Cumulative stakes create monopolies. A signal that was great
3 years ago but mediocre now retains a dominant stake from compounded
growth, blocking new signals from gaining influence. Real markets
don't work this way:

- **Numerai**: models scored on rolling 20-round window. Past
  glory doesn't help.
- **Real markets**: ongoing costs (fees, salaries, rent) create
  constant pressure. Investors redeem based on recent performance.
- **Polymarket**: each bet is independent. No accumulated advantage.

Rolling window ensures all signals compete on recent merit. A signal
must continuously justify its existence. New signals have a fair
chance against incumbents.

#### Properties

- **New signals start fair.** Evaluated on the same rolling window
  as everyone else. No permanent disadvantage vs incumbents.
- **Valuable signals earn weight.** Consistent recent marginal
  contribution → high stake → high influence.
- **Fading signals lose weight.** Edge decays → rolling marginal
  drops → stake shrinks → less influence.
- **Redundant signals die.** Zero marginal contribution →
  zero stake → removal.
- **Harmful signals die fast.** Negative marginal contribution →
  negative stake → immediate removal.

This is analogous to:
- Numerai MMC: scored on what you add to the meta-model
- Polymarket: accurate predictors profit, inaccurate ones lose
- Real markets: informed participants accumulate capital

### Risk Management

Portfolio-level, separate from signal evaluation:

- **Drawdown circuit breaker**: if portfolio drawdown exceeds threshold
  (e.g. 10%), reduce all positions to a fraction of normal. Gradual
  recovery after drawdown heals.
- **Hard stop**: extreme drawdown (e.g. 20%) → halt trading, require
  manual review.
- **Position limits**: maximum total exposure, per-asset limits.

These protect the portfolio from systemic events that individual signal
scoring cannot anticipate.

### 5. Trade

Portfolio weights from the stake system determine position sizes.
Each signal's horizon determines rebalancing frequency:

- horizon=1: daily rebalance
- horizon=5: rebalance every 5 days
- horizon=20: rebalance every 20 days

Mixed horizons provide natural time-scale diversification.

**Portfolio-level evaluation uses Sharpe, drawdown, costs.**
This is the only place Sharpe appears in the pipeline.

### Paper → Real

The entire system runs on paper first. Graduation to real capital is
a portfolio-level decision, not per-signal:

- Paper portfolio Sharpe > threshold for N consecutive days
- No individual signal graduates independently
- Real capital allocation is proportional to paper track record

## Architecture: Producer-Consumer Separation

The current system is tightly coupled: each daemon loads data, computes
signals, evaluates them, manages the registry, and executes trades. This
means adding a new signal type (ML, classical indicator, external) requires
changes throughout the codebase.

The target architecture separates production from consumption:

```
Producers (independent, know nothing about the pipeline):
  ├─ GP daemon      — explores DSL expression space
  ├─ ML daemon      — trains and runs ML models
  ├─ Classical job   — computes RSI, MACD, carry, momentum, etc.
  ├─ External ingest — reads from APIs, prediction markets, etc.
  └─ Human CLI       — manual submission

         │
         │  each writes: (signal_id, date, prediction_value, asset)
         ▼

Prediction Store (the only coupling point):
  "signal X predicted Y on date Z for asset A"
  Simple append-only table. No computation, no evaluation.

         │
         │  pipeline reads predictions + realized outcomes
         ▼

Consumer (single pipeline, knows nothing about producers):
  reads predictions → scores IC → updates stakes → portfolio weights → trade
```

### Why this matters

**Adding a new signal type = writing a new producer.** The producer
computes predictions and writes them to the store. It doesn't import
the pipeline, the registry, the DSL, or any evaluation code. The
pipeline doesn't change.

**The pipeline is truly output-only.** It reads predictions from the
store. It doesn't know if a prediction came from a DSL expression, an
ML model, or a human. It can't inspect internals because it has no
access to the producer code.

**signal-noise stays focused.** It collects raw market data (prices,
macro, on-chain). It does NOT compute signals, run models, or store
predictions. That's alpha-os's job.

### Prediction Store schema

```sql
CREATE TABLE predictions (
    signal_id   TEXT NOT NULL,
    date        TEXT NOT NULL,
    asset       TEXT NOT NULL,
    value       REAL NOT NULL,
    horizon     INTEGER NOT NULL DEFAULT 1,
    recorded_at REAL NOT NULL,
    PRIMARY KEY (signal_id, date, asset)
);
```

Each producer writes to this table. The pipeline reads from it.
Signal metadata (source, expression, model path) is stored in a
separate `signals` table for reproducibility and auditing, but is
never read by the evaluation pipeline.

### Signal types

Different producers, same output format.

| Producer | What it does | Writes to prediction store |
|----------|-------------|---------------------------|
| GP daemon | Explores DSL space, finds novel patterns | DSL evaluation output |
| Classical job | Computes RSI, MACD, carry, momentum | Indicator values |
| ML daemon | Trains models, runs inference | Model predictions |
| External ingest | Reads prediction markets, APIs | External values |
| Human CLI | Manual rules, domain knowledge | Manual values |

### Migration path

Incremental migration from monolithic to producer-consumer.

1. ~~Create the prediction store~~ — done (`predictions.db`)
2. ~~Make GP daemon write predictions to the store~~ — done (506 signals)
3. ~~Registry producer: evaluate active alphas → store~~ — done (638 signals)
4. ~~Trader reads from prediction store~~ — done (fallback to direct eval)
5. ~~Add a classical indicator producer~~ — done (7 signals: RSI, zscore, carry, dollar, F&G, vol, gold-dxy)
6. Remove direct coupling between producers and pipeline
7. ~~Remove internal inspection code (feature caps, semantic dedup)~~ — done

## Diversity

Diversity is maintained at two layers without explicit diversity rules.

**Generation: MAP-Elites.**
The discovery pool uses a behavioral grid (persistence × activity ×
price_beta × vol_sensitivity) to ensure candidates are structurally
diverse. GP cannot converge to a single expression pattern — the grid
forces exploration of different behavioral niches.

**Selection: Marginal contribution.**
Correlated signals have near-zero marginal contribution. If 10 signals
all produce similar output, only the first adds value — the other 9
stagnate and die. The portfolio naturally diversifies without inspecting
signal internals.

**Structural diversity axes:**
- Horizons (1, 5, 20 days)
- Prediction targets (returns, volatility, cross-sectional)
- Asset classes (crypto, stocks, ETFs, commodities)
- Generator types (DSL/GP, ML, future additions)

These multiply combinatorially, creating a large diversity space that
the pipeline explores through generation and selects through marginal
contribution.

**Remaining risk: feature concentration.**
If all high-stake signals depend on the same underlying feature
(e.g. `fear_greed`), the portfolio is exposed to that feature breaking.
The output-only principle prevents direct feature inspection. However,
marginal contribution mitigates this indirectly: same-feature signals
produce correlated output → low marginal → limited concentration.
Complete mitigation requires feature-diverse generation (MAP-Elites
behavioral dimensions help here).

## Pipeline Validation

The pipeline itself (not individual signals) must be validated.

**Historical simulation**: run the full pipeline on historical data.
Start from zero signals, generate, admit, simulate stake updates with
realized returns, measure portfolio Sharpe/drawdown.

This tests whether stake-based selection, multi-asset combination,
and capital allocation work as a system. It is a backtest of the
mechanism, not of individual signals.

**Caution**: tuning pipeline parameters (reward rate, min_stake,
max_weight) on historical data risks overfitting the pipeline itself.
Keep parameters minimal and driven by design principles, not
optimization.

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

## Implementation Progress

1. ~~Multi-horizon forward returns in `evaluate_cross_asset`~~ — done
2. ~~Residualized returns in IC path~~ — done
3. ~~Store horizon in discovery pool schema~~ — done
4. ~~Rewrite admission to validate OOS IC~~ — done
5. ~~Generator fitness_metric hardcoded to IC~~ — done
6. ~~Stake column + bootstrap + daily rolling update~~ — done
7. ~~Portfolio weights from stakes (TC fallback)~~ — done
8. ~~Remove feature cap, semantic dedup, state transitions~~ — done
9. ~~Remove IC→Sharpe fallback shims~~ — done
10. ~~Prediction store + GP producer~~ — done (506 signals)
11. ~~Registry producer: active alphas → store~~ — done (638 signals)
12. ~~Trader reads from prediction store (fallback to direct eval)~~ — done
13. ~~Classical indicator producer~~ — done (7 signals, 3rd producer)
14. ~~Full marginal contribution scoring (leave-one-out)~~ — done
15. ~~Remove prune overflow, dead admission code~~ — done (322 lines deleted)

## Deprecation Schedule

Legacy code that will be removed as the stake system matures.

### Remove when stakes are live (~60 days of forward returns)

| Code | File | Why wait |
|------|------|----------|
| `_prune_active_overflow` (cap=600) | `admission.py` | Memory/compute guard until stake-based removal works |
| `active_quality_min` / `dormant_revival_quality` config | `config.py` | Thresholds no longer used but config fields remain |

### Remove when producer-consumer is implemented

These are structural changes that require the prediction store.

| Code | File | Why wait |
|------|------|----------|
| Direct `parse(expr).evaluate(data)` in admission | `admission.py` | Admission should read from prediction store, not compute |
| Direct `parse(expr).evaluate(data)` in trader | `trader.py` | Trader should read from prediction store |
| `BacktestEngine` in admission | `admission.py` | PBO still uses it; replace with IC-only validation |
| `fitness_metric` config field | `config.py` | Used by trader/lifecycle; remove after full stake migration |
| `AlphaState` enum | `managed_alphas.py` | Only after all consumers use stake instead of state |

### Already removed

| Code | When | Commit |
|------|------|--------|
| Feature cap check | 2026-03-21 | `341a55d` |
| Semantic dedup check | 2026-03-21 | `341a55d` |
| State transitions in lifecycle | 2026-03-21 | `341a55d` |
| State-based trading filter → stake > 0 | 2026-03-21 | `341a55d` |

### Low-risk cleanup (remove when convenient)

| Code | Notes |
|------|-------|
| `oos_sharpe_min` in `GateConfig` construction (admission) | Already hardcoded to 0.0 |
| `GateTomlConfig.oos_log_growth_min` etc. | IC admission doesn't use these |
| `_OOS_FITNESS_MAP` fallback for "ic"/"ric" | Temporary compat shim |

## Constraints

- Memory: 7.6GB server limits concurrent data loading
- eval universe cached to avoid recomputation
- IC computation is cheap (rank correlation), enabling high throughput
- ML integration is future work; pipeline designed to accommodate it
- Bootstrap period: ~40 more days until stakes reflect live performance
