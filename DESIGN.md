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
  a living thing. Mutation creates offspring within the same feature subset.
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
  Admission Gate ─── fail ──→ REJECTED
      │
      pass
      ▼
  CANDIDATE ──────── pass ───────→ ACTIVE
      │                              ▲
      │ fail                         │ revival
      ▼                              │
  REJECTED                     DORMANT
```

The pipeline runs as a continuous cycle: generate candidates, validate
statistically, admit survivors, then monitor and demote as edges decay.

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
- **Mutation** (30%): swap feature, change window, or replace operator
- **Selection**: tournament (size=3) with elitism
- **Bloat control**: fitness penalty of 0.01 × node_count, max depth=3
- **Feature subsets**: each evolution round uses a random subset of K=27
  features (√753 ≈ 27), ensuring diversity across alphas
- **Quality-diversity**: MAP-Elites archive with 3D behavior descriptor
  (feature_bucket mod 100, holding half-life, complexity) → 10,000 cells

Crossover was removed: it is incompatible with feature subsets (parent
expressions reference different feature sets) and contributed little value
with large populations of similar alphas.

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

Current runtime states are `candidate`, `active`, `dormant`, and `rejected`.
Historical `born` and `probation` rows are normalized to `candidate` and
`active` on read.

| Transition             | Condition |
| ---------------------- | --------- |
| CANDIDATE → ACTIVE     | Passes the admission gate (`candidate_quality_min`, PBO, DSR, correlation) |
| CANDIDATE → REJECTED   | Fails the admission gate |
| ACTIVE → DORMANT       | Blended quality < `active_quality_min` |
| DORMANT → ACTIVE       | Blended quality ≥ `dormant_revival_quality` and enough forward observations |

Runtime quality is blended from historical OOS quality and forward returns.
The confidence weight rises with the number of forward observations, so new
alphas shrink toward their historical prior instead of being treated as zero.

## Trading Universe

The registry is not the live trading universe.

- `alphas.state=active` means an alpha is eligible for deployment.
- `trading_universe` is the explicitly deployed subset that the trade runtime reads.
- `refresh-universe` populates that subset using blended quality, slot count,
  replacement limits, and a promotion margin.

This keeps research churn (`candidate` admission and lifecycle updates) separate
from the set that actually drives positions.

## Runtime Layer Boundaries

The trading runtime should separate prediction from portfolio decisions and
venue execution. This is an explicit design goal, not just an implementation
detail.

### Why Separation Matters

When these concerns are mixed, small prediction changes leak directly into
order placement, and venue constraints become the place where strategy errors
are discovered. That creates avoidable churn:

- tiny target changes become micro-orders that cannot clear venue minimums
- exchange-specific constraints distort strategy behavior late in the pipeline
- replay experiments become less realistic because prediction and execution
  assumptions are coupled

The runtime should reject untradeable intents before they reach the executor.

### Intended Runtime Stack

```
Prediction
    ↓
Portfolio Construction
    ↓
Execution Planning
    ↓
Venue Constraints
    ↓
Execution
```

### Responsibilities

| Layer | Responsibility | Examples |
| ----- | -------------- | -------- |
| Prediction | Produce alpha-level and combined signals | alpha scoring, blended quality, consensus signal |
| Portfolio Construction | Convert signals into desired holdings | target position, risk scaling, trading deadband |
| Execution Planning | Convert holdings gap into order intents | side, qty delta, urgency, split preference |
| Venue Constraints | Make intents executable on a specific venue | min notional, lot size, precision, fee-aware rounding |
| Execution | Submit and reconcile executable orders | retries, optimizer delays, fills, reconciliation |

### Canonical Runtime Objects

The runtime should converge on three explicit handoff objects:

- `TargetPosition`
  - The desired post-trade holding for an asset.
- `ExecutionIntent`
  - The delta between current holdings and the target position.
- `ExecutableOrder`
  - A venue-valid order that already satisfies precision and notional rules.

Executors should only accept `ExecutableOrder`-level inputs. They should not be
responsible for deciding whether a trade is economically meaningful.

### Current Refactoring Direction

The codebase is moving toward this separation in stages:

1. Keep research churn in the registry and deployment churn in `trading_universe`.
2. Stop sending sub-minimum or low-value rebalances downstream.
3. Move deadband and minimum-trade decisions into portfolio construction and
   venue-constraint layers.
4. Keep the executor focused on venue interaction, not strategy cleanup.

This boundary is now part of the intended architecture. Future refactors should
prefer clearer layer ownership over parameter-only tuning.

## Runtime Complexity Review

The runtime has become more structured, but it has also become more complex.
That complexity is not all the same.

### Complexity to Keep

These changes add moving parts, but they reduce coupling and should remain:

- `trading_universe` separated from the research registry
- explicit runtime handoffs (`TargetPosition`, `ExecutionIntent`, `ExecutableOrder`)
- shared runtime cost model across replay, paper, and exchange execution

These are architecture boundaries, not incidental knobs.

### Complexity to Reduce

These parts currently add operational complexity without enough return:

- execution optimizer hard-block rules (`wait`, retry, then execute anyway)
- circuit breaker state that survives strategy/runtime regime changes unchanged
- strategy policy still spread across `trade`, optimizer policy, and runtime guards
- signal-source behavior that is not transparent enough (`latest` vs recent data)

The main smell is not "too many modules". The main smell is too many boolean
runtime gates that delay or block execution, then fall back to execution anyway.

### Simplification Priority

The preferred simplification order is:

1. Remove or weaken optimizer hard blocks in favor of softer penalties or size reduction.
2. Make the circuit breaker strategy-aware so new runtime regimes do not inherit old loss streaks blindly.
3. Move more runtime policy out of `trade` into explicit policy objects with narrow responsibilities.
4. Keep signal freshness semantics explicit and observable in logs and reports.

Short version: keep structural separation, reduce policy branching.

### Complexity Budget

New runtime work should stay within a small complexity budget:

- do not add new alpha lifecycle states
- do not add new long-lived pools beyond the registry and `trading_universe`
- do not add new tuning knobs when an existing quality threshold or slot limit can solve the problem
- prefer hard caps and one-in/one-out replacement over new scoring layers
- remove migration fallbacks after the new path is proven in testnet

The preferred fix for registry growth is therefore a simple active-cap rule,
not a new shadow pool or a second deployment lifecycle.

### Current Registry Cap

The active BTC registry now uses a hard cap through
`admission.max_active_alphas`.

- the cap applies only to `alphas.state=active`
- stronger incoming alphas can replace weaker incumbents
- overflow is resolved by demoting weakest active rows to `dormant`
- `trading_universe` remains separate and is still refreshed explicitly

This keeps the control mechanism simple: one cap, one demotion path, no new
runtime state.

The same bias applies to evaluation: use short observation windows after
material runtime changes. The goal is a fast go / no-go decision, not a long
freeze period with unclear attribution.

### Current Evaluation Posture

The current BTC testnet profile is not in an open-ended optimization phase.
It is in a short observation phase after several structural changes:

- deployed `trading_universe`
- runtime cost model
- deadband-based small-trade suppression
- simpler execution optimizer
- strategy-aware circuit breaker
- capped active registry growth

This means:

- the system is ready to measure operational behavior now
- the system is not yet ready to claim durable profitability
- the next decision should be made from a short observation window, not from
  another round of immediate runtime tuning

During this phase, documentation, observability, and operational hygiene are
in-scope. New strategy complexity is not.

## Admission Gate

All criteria must pass for a candidate to be admitted:

| Check         | Threshold | Notes |
| ------------- | --------- | ----- |
| OOS Sharpe    | `candidate_quality_min` | Purged Walk-Forward CV (5 folds) |
| PBO           | `pbo_max` | Batch-computed |
| DSR p-value   | `dsr_pvalue_max` | Batch-computed |
| Correlation   | `correlation_max` | Avg correlation with registry-active alphas |
| Min days      | ≥ 200     | Minimum data requirement |

In practice the BTC testnet profile currently runs a much stricter
`candidate_quality_min` than the default config. The runtime lifecycle then
handles post-admission quality control via blended quality and dormant revival.

## Signal Combination: Quality × Diversity Weighting

The runtime does not combine every registry alpha directly. It uses the
deployed trading universe and then applies a
three-stage selection pipeline:

1. Start from deployed `ACTIVE` alphas in `trading_universe`.
2. Preselect a larger set, then rerank by blended quality and confidence.
3. Apply a correlation filter to cap the final selected set.

Each selected alpha's weight reflects both its blended quality and
its uniqueness relative to the portfolio (diversity):

```
weight_i = max(blended_quality_i, 0) × diversity_i + min_weight
```

Where:
- **Quality** = `max(blended_quality, 0)` — weak alphas get minimal
  weight but are not excluded entirely (`min_weight` floor).
- **Diversity** = `1 - mean(|corr(i, j)|)` for all j ≠ i — alphas
  uncorrelated with the rest contribute more.
- **min_weight** = 1e-4 — ensures no alpha is fully zeroed out.

Diversity scores are recomputed every 63 days using a 252-day lookback
window. The final correlation filter currently caps the selected set at
`max_trading_alphas` (default 30) with `max_correlation=0.3`.

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
active alpha lifetime is ~1.6 days before dormancy or replacement. This means
signal_consensus measures agreement among a different set of alphas each
cycle, reducing its reliability as a sizing signal.

**Root cause**: The problem is not generation rate (144k/day) but
ease of displacement. New alphas are GP-fitted to "data up to today"
and naturally score higher on backtest metrics than incumbents fitted
days ago. The ranking is based solely on backtest metrics, so freshly
overfit alphas always displace older ones — overfitting recency, not
genuine improvement. Potential Path A mitigations: forward performance
in ranking, tenure bonus, incumbent advantage (hysteresis), adoption
rate caps. Path B treats displacement as a given.

Two independent approaches are being considered:

- **Path A (stabilize)**: Reduce generation rate, raise adoption bar,
  add tenure bonus to lifecycle scoring — make the top 30 stable so
  consensus becomes meaningful. This improves the current alpha-os.
- **Path B (turnover-native)**: Build a separate system designed for
  high-turnover alpha statistical voting. Instead of relying on a stable
  top 30, treat all alphas as ephemeral voters.

These are independent: Path A improves alpha-os as-is, Path B is a
separate system for a different paradigm. They are not layered.

### Path B: Design Discussion (2026-03-04)

#### Direction vs Distribution

The current system (consensus) and the initial Path B design (voting)
are both **directional prediction** — "do more alphas say long or
short?" This is the weakest form of signal combination:

- With micro-returns (±0.02%), direction easily flips on noise
- Win rate alone doesn't determine profitability — a 80% win rate
  with asymmetric losses loses money; a 20% win rate with asymmetric
  gains makes money
- The question should not be "up or down?" but "what does the return
  distribution look like?"

The right approach is **distributional** — design the portfolio's
return distribution (mean, variance, skewness, tail behavior) rather
than predict direction. However, distributional methods (Kelly, CVaR)
were removed because they require stable historical data, which
short-lived alphas cannot provide (see "Design Rationale" above).

#### Why Voting Fails: The 4 Conditions

Prediction markets (Polymarket, etc.) work because they satisfy the
four conditions for "wisdom of crowds" (Surowiecki / Condorcet):

1. **Diversity** — voters hold different viewpoints
2. **Independence** — voters are not influenced by each other
3. **Decentralization** — each voter has access to unique information
4. **Aggregation** — a mechanism to properly combine individual votes

Alpha-os currently violates all four:

| Condition | Prediction Market | Alpha-OS |
|---|---|---|
| Diversity | Each person has unique analysis | Same GP, same feature pool |
| Independence | Separate brains, no coordination | Crossover creates parent-child lineages |
| Decentralization | Each person has private info | All alphas read the same 753 signals |
| Aggregation | Price discovery via real money | Naive weighted average |

With 25,000+ alphas generated from the same GP and the same data,
voting is essentially **one opinion echoed 25,000 times**. Adding
more voters adds noise, not information.

#### Solving the 4 Conditions

| Condition | Difficulty | Approach |
|---|---|---|
| Aggregation (4) | Easy | Code design — replace naive average with cluster-representative voting |
| Independence (2) | Medium | Cluster similar alphas → 1 vote per cluster, reducing 25,000 to ~50 independent voices |
| Diversity (1) | Medium-Hard | Partition GP into sub-populations with different operator sets or evolution pressures |
| Decentralization (3) | Hard | Assign each alpha a **feature subset** (e.g. macro-only, technical-only, sentiment-only) so they see different slices of reality. 753 signals across 6 domains exist; the issue is all alphas can access all of them |

Conditions 2 and 4 are solvable with current architecture. Condition 1
requires GP generation changes. Condition 3 is the hardest — it
requires restricting each alpha's feature access, which changes the
fundamental GP design.

#### Design Decisions (2026-03-05)

The following decisions resolve conditions 1 and 3 simultaneously,
and simplify GP by removing crossover.

**Decision 1: Random feature subsets (conditions 1 + 3)**

Each alpha is born with a random subset of K features from the
available 753. This is analogous to Random Forest feature bagging:

- Each alpha sees a different slice of reality → decentralization (3)
- Different inputs produce different strategies → diversity (1)
- No manual domain partitioning needed — randomness handles it
- Proven approach (Random Forest, 2001; Breiman)

```
Alpha generation:
  1. Sample K features uniformly from 753 available
  2. Generate random expression tree using only those K features
  3. Mutation operates within the same feature subset
  4. Feature subset is fixed for the alpha's lifetime
```

Open parameters:
- K: how many features per alpha? √753 ≈ 27 as starting point
- Whether to include a mandatory feature (e.g. btc_ohlcv) — TBD

**Decision 2: Remove crossover (simplification)**

Crossover and feature subsets are incompatible:

```
Parent A subset: {btc_ohlcv, vix, cpi}
Parent B subset: {gold, sp500, pmi}
Child expression references features from both parents
→ Child's subset must be the union → subsets grow over generations
→ After N generations, all alphas use all features → diversity lost
```

Alternatives (inheriting one parent's subset, re-sampling for children)
all add complexity. The simplest solution: **remove crossover entirely**.

- Mutation-only GP is well-established (Cartesian GP, etc.)
- Crossover was likely contributing little value in current alpha-os:
  with 25,000 similar alphas, crossover recombines near-identical
  subtrees — it generates variety in form but not in substance
- Removing crossover also eliminates bloat (tree size inflation)

Evolution becomes: **random generation → mutation → selection**.

**Decision 3: Deprecate recency weighting**

The initial voting design weighted newer alphas higher (recency_weight).
This is flawed:

- "New = good" has no theoretical basis
- Newest alphas have the least accuracy data — the highest-weighted
  voters are the least trustworthy
- GP generates for past-data fit, not market adaptation — newness
  does not correlate with current-market relevance

Recency weighting is removed from the voting design. Voter weight
will be determined by other factors (accuracy, cluster membership)
once conditions 2 and 4 are resolved.

#### Remaining Work

| Condition | Status | Next Step |
|---|---|---|
| 1. Diversity | **Decided**: random feature subsets | Implement feature subset in GP |
| 2. Independence | **Deferred**: not needed initially | Feature subsets provide implicit decorrelation |
| 3. Decentralization | **Decided**: random feature subsets | Same implementation as condition 1 |
| 4. Aggregation | **Decided**: MAP-Elites distributional sizing | Resolve open design questions, then implement |

Implementation order: **1+3 (feature subsets in GP) → behavior
descriptor redesign → signal normalization → sizing formula →
integration**.

Conditions 2 and 4 require condition 1+3 to be in place first — without
diverse alphas, clustering and aggregation operate on redundant data.

#### Ensemble Distributional Forecasting

Path B's sizing problem: how to determine position size, not just
direction? Traditional approaches (Kelly criterion, CVaR) require
stable historical return distributions — impossible when alphas live
1.6 days and past returns reflect dead alphas (see Position Sizing
section: "No Kelly criterion", "No CVaR gate").

**Key insight**: instead of individual alphas estimating return
distributions (impossible with short lifespans and insufficient data),
use the **distribution of all alpha signals** as the predictive
distribution. This is analogous to Random Forest uncertainty
quantification — individual trees are weak predictors, but the
ensemble's prediction spread captures uncertainty.

```
25,000 alpha signals → signal distribution
                       ├── μ (mean)      → directional bias
                       ├── σ² (variance) → disagreement = uncertainty
                       ├── skewness      → tail risk asymmetry
                       └── tail density  → extreme event probability
```

Why this works:
- **Real-time**: uses current signal values, no history dependency
- **Self-calibrating**: more diverse alphas (condition 1+3) →
  broader distribution → better uncertainty quantification
- **No dead-alpha problem**: only living alphas contribute
- **Scale-independent**: works whether there are 100 or 100,000 alphas

Alpha signals are not returns, but the **shape** of the signal
distribution provides distributional information for sizing:

| Signal distribution shape | Interpretation | Sizing response |
|---|---|---|
| Tight, one-sided (low σ, high μ) | Strong consensus | Larger position |
| Wide, centered (high σ, low μ) | Maximum disagreement | Minimal or no position |
| Skewed left | Downside tail risk | Reduce or hedge |
| Bimodal | Two distinct regimes | Reduce (ambiguity) |

This approach is known in the literature as **ensemble distributional
forecasting** or **uncertainty quantification via ensemble**. The
ensemble's prediction distribution serves as a proxy for the true
predictive distribution of returns.

Note: the exact sizing formula (how μ, σ, skewness map to position
size) is condition 4 (aggregation) and awaits design.

#### Architecture: MAP-Elites + Distributional Sizing

Path B uses the MAP-Elites archive as the sole source of trading
signals. Instead of adopting a top-30 into a registry with lifecycle
management, the entire archive population is evaluated each cycle
and the signal distribution determines position sizing.

**Signal flow (two-level aggregation)**:

```
GP evolution (continuous, mutation only, no crossover)
  ├── each individual born with K random features from 753
  ├── sanity filter: non-constant, NaN < 10%, finite values
  └── candidates → MAP-Elites archive (cell = behavior descriptor)

Trade cycle:
  Level 1 — per cell (sign normalization):
    archive elites → evaluate on current data → sign(signal) → {-1, +1}
    per cell: long_pct = count(+1) / count(votes in cell)

  Level 2 — across cells (distributional sizing):
    cell_long_pcts = [cell_A.long_pct, cell_B.long_pct, ...]
    μ_cells     = mean(cell_long_pcts)
    σ_cells     = std(cell_long_pcts)
    skew_cells  = skewness(cell_long_pcts)

  Sizing:
    direction  = sign(μ_cells - 0.5)
    confidence = |μ_cells - 0.5| × 2 / (|μ_cells - 0.5| × 2 + σ_cells)
    skew_adj   = clip(1 - |skew_cells| × k, 0.5, 1.0)
    position   = direction × confidence × skew_adj × dd_scale × max_pos × portfolio
```

Sign normalization solves the scale problem (DSL expressions produce
wildly different scales). Two-level aggregation preserves distributional
information: Level 1 reduces each cell to a vote, Level 2 examines
how votes are distributed across behavioral niches. When all niches
agree (low σ), confidence is high. When niches disagree (high σ) or
the disagreement is asymmetric (high |skewness|), position shrinks.

This is structurally identical to Random Forest classification:
each tree (alpha) votes, but the ensemble's confidence comes from
the distribution of votes across diverse trees (feature subsets).

**Why MAP-Elites, not particle filter**: Particle filters suffer from
particle degeneracy — after a few steps of resampling, nearly all
particles descend from a single ancestor and diversity collapses.
MAP-Elites avoids this by design: each cell holds exactly one elite,
and individuals in different cells never compete. Diversity is a
structural guarantee, not a tunable parameter.

**Correlation between alphas**: Feature subsets ensure that alphas
in different cells use different information sources. If all alphas
agree despite using different features, this is genuine consensus
(Wisdom of Crowds), not echo chamber. No explicit correlation
management is needed beyond feature subsets.

**Comparison with current system (Path A)**:

| | Path A (current) | Path B (MAP-Elites) |
|---|---|---|
| Evolution | crossover + mutation | mutation only |
| Features | all 753 available | K random per individual |
| Management | registry + lifecycle | MAP-Elites archive only |
| Signal source | top-30 from registry | all archive elites |
| Sizing | consensus = \|μ\|/(|μ|+σ) | two-level: sign → per-cell vote → cross-cell distribution |
| Correlation | quality × diversity weights | feature subsets (implicit) |
| Alpha turnover | problem (1.6 days) | irrelevant (archive is the model) |

**Root cause analysis**: Alpha turnover is caused by ease of
displacement, not generation rate. New alphas are GP-fitted to the
latest data and naturally outscore incumbents on backtest metrics.
Path A mitigates this (tenure bonus, hysteresis). Path B eliminates
the concept entirely — there is no "adoption" or "retirement", only
archive cells being updated when a better individual arrives.

#### Open Design Questions

| # | Question | Notes |
|---|---|---|
| 1 | **Fitness function** | **Decided**: sanity filter only (non-constant, NaN < 10%, finite values). No performance-based fitness. With single-asset BTC, forward performance cannot distinguish skill from luck (one binary outcome per cycle for 10,000 alphas). Individual alpha quality is irrelevant — ensemble power comes from diversity, not individual fitness (Random Forest analogy: individual tree accuracy is not optimized). |
| 2 | **Behavior descriptor redesign** | **Decided**: 3 axes — feature_bucket (hash(feature_set) % 100), holding_half_life (10 bins), complexity (10 bins). Total = 100 × 10 × 10 = 10,000 cells. Feature hash directly reflects feature subset diversity without requiring domain labels. Same feature set → same cell → within-cell competition. Different features → different cell → coexistence guaranteed. Replaces corr_to_live_book (Path A, needs registry) and turnover (redundant with half_life). |
| 3 | **Signal normalization** | **Decided**: sign(signal) → {-1, +1}. Completely eliminates scale problem. Magnitude information is unnecessary — ensemble power comes from diversity of opinions, not strength of individual conviction. Two-level aggregation (per-cell vote → cross-cell distribution) preserves distributional information despite binary individual signals. |
| 4 | **K value for feature subsets** | Starting point √753 ≈ 27. Too small → alphas are too constrained. Too large → alphas overlap too much, losing diversity. Needs experimental tuning. |
| 5 | **Archive dimensions** | **Decided**: 100 (feature_bucket) × 10 (half_life) × 10 (complexity) = 10,000 cells. Resolved together with behavior descriptor redesign (#2). |
| 6 | **skew_adj parameter k** | Initial value 0.5 (full penalty at \|skewness\| ≥ 1.0). Whether to penalize both directions of skew or only counter-directional. Now applies to cross-cell long_pct distribution skewness, not raw signal skewness. |

#### Implementation Plan

```
Phase 1 (parallel):
  ├── Step 1: GP feature subsets + crossover removal          ✅
  │     generator.py: with_random_subset(), feature_subset param
  │     gp.py: mutation-only evolution, generator injection
  │
  ├── Step 2: New behavior descriptor                         ✅
  │     behavior.py: feature_hash(100) × half_life(10) × complexity(10)
  │     archive.py: new ArchiveConfig for 3-axis grid
  │
  ├── Step 3: Sanity filter                                   ✅
  │     archive.py: add_if_empty() with passes_sanity_filter()
  │     (non-constant, NaN < 10%, finite values)
  │
  └── Step 4: Two-level aggregation                           ✅
        voting/ensemble.py: per-cell sign vote → cross-cell distribution
        → confidence, skew_adj, direction calculation

Phase 2:
  ├── Step 5: Trader integration                              ✅
  │     trader.py: combine_mode="map_elites" path
  │     config.py + default.toml: Path B parameters
  │     default remains "consensus" (Path A unchanged)
  │
  └── Step 6: Evo daemon MAP-Elites mode                      ✅
        daemon/evo.py: evo_mode="map_elites" round with feature subsets
        archive.py: SQLite persistence (save_to_db / load_from_db)
```

All steps completed with tests passing (508 tests). Each step is one
commit. Default config remains "consensus" / "legacy" (Path A unchanged).

#### Status

**Implementation complete** (2026-03-05). Key files:
- `src/alpha_os/dsl/generator.py` — `with_random_subset()` for feature bagging
- `src/alpha_os/evolution/gp.py` — mutation-only GP, generator injection
- `src/alpha_os/evolution/behavior.py` — 3D behavior descriptor (feature_bucket, half_life, complexity)
- `src/alpha_os/evolution/archive.py` — MAP-Elites grid + `add_if_empty()` + SQLite persistence
- `src/alpha_os/voting/ensemble.py` — two-level ensemble aggregation
- `src/alpha_os/paper/trader.py` — `combine_mode="map_elites"` path
- `src/alpha_os/daemon/evo.py` — `evo_mode="map_elites"` round with archive persistence

## Paper Trading

Two modes:

- **Daily cycle** (`paper --once`): Syncs data, evaluates all alphas,
  applies lifecycle transitions, combines signals with quality × diversity
  weights, adjusts for risk (drawdown + volatility scaling), and executes
  via paper executor.
- **Historical replay** (`paper --replay`): Vectorized historical
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

#### Phase 3: CLI `trade` Command (1 day) ✅

Based on existing `paper` command:

```bash
# Testnet (default — no real money)
python3 -m alpha_os trade --once

# Production (real money, explicit flag required)
python3 -m alpha_os trade --once --real --capital 1000

# Scheduled daily execution
python3 -m alpha_os trade --schedule
```

Reuses PaperTrader's full cycle (data sync → alpha eval → combination →
risk adjustment → execution) with BinanceExecutor instead of PaperExecutor.

#### Phase 4: Testnet Validation (1-2 weeks)

Run `trade --schedule` daily on Binance testnet:

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
  ├── run_trade.sh --asset BTC
  ├── run_trade.sh --asset ETH
  └── run_trade.sh --asset SOL
```

Implementation:
- Add per-asset DB isolation (`alpha_registry_ETH.db`, etc.) or
  asset column in existing tables
- Extend `run_trade.sh` to accept `--asset` parameter
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
│ evo daemon  │  │ admission   │  │ trade runtime │  │   lifecycle    │
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
2. **admission daemon** reads PENDING candidates, runs statistical validation
   (purged WF-CV, DSR, PBO, admission gate), computes incremental diversity
   against existing ACTIVE alphas, and writes to `alphas` (ACTIVE) and
   `diversity_cache`.
3. **trade runtime** reads the deployed `trading_universe` and diversity cache, computes a
   shortlist, applies correlation filtering and risk adjustments, and
   executes via Binance. No evolution, no lifecycle evaluation.
4. **lifecycle manager** reads forward returns, computes rolling Sharpe
   (63-day window), and transitions alpha states (ACTIVE ↔ DORMANT).
5. **refresh-universe timer** runs after lifecycle, updates the deployed
   `trading_universe`, and keeps live trading on a slower replacement cadence.

### Process Definitions

#### evo daemon (continuous)

Responsibility: continuously explore the alpha search space and feed
candidates to the admission daemon.

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

#### admission daemon (triggered batch)

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
  5. Admission gate (OOS Sharpe, PBO, DSR, correlation, min_days).
- Candidates that pass: `registry.register()` as ACTIVE, update
  `candidates.status = 'adopted'`.
- Candidates that fail: update `candidates.status = 'rejected'`.

**Incremental diversity computation** (key optimization):

Instead of computing the full N×N correlation matrix, the admission daemon
computes correlations only between new candidates and existing ACTIVE
alphas. For a batch of 100 new candidates against 50,000 ACTIVE alphas,
this is 5M pairs instead of 2.5 billion — a 500× reduction.

Results are written to the `diversity_cache` table. The trade cycle reads
this cache directly instead of recomputing diversity.

Implementation: `src/alpha_os/daemon/admission.py`

CLI entry point:

```
python -m alpha_os admission-daemon --asset BTC [--config ...]
```

#### trade cycle (periodic 4h) — existing Trader, slimmed down

Responsibility: combine active alphas, apply risk adjustments, execute.

This is the existing `Trader.run_cycle()` (paper/trader.py) with two
responsibilities removed:

1. **Lifecycle evaluation** (monitor.check → lifecycle.evaluate → state
   transitions): moved to lifecycle manager.
2. **Diversity recomputation** (_recompute_diversity): moved to admission daemon.

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

No new CLI command — uses existing `trade --schedule`.

#### lifecycle manager (daily)

Responsibility: evaluate forward performance of all alphas and transition
states.

Extracts the lifecycle evaluation loop from `Trader.run_cycle()` lines
304-336 and the equivalent logic in `ForwardRunner.run_cycle()`.

- Runs daily at UTC 00:30 via systemd timer.
- Reads all ACTIVE and DORMANT alphas from registry.
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

Updated by the admission daemon when new alphas are adopted. Read by the trade runtime
for weight computation.

#### WAL mode for all databases

Currently only `DataStore` (alpha_cache.db) enables WAL mode. Extend to
all databases for multi-process safety:

| Database | WAL | busy_timeout | Writer(s) | Reader(s) |
|----------|-----|-------------|-----------|-----------|
| alpha_cache.db | ✅ existing | 30s | trade cycle (sync) | evo daemon |
| alpha_registry.db | **NEW** | 30s | evo (candidates), admission (alphas, div_cache), lifecycle (alphas.state) | trade runtime |
| forward_returns.db | **NEW** | 30s | trade cycle (record) | lifecycle |
| paper_trading.db | **NEW** | 30s | trade cycle | — |

Write contention is limited to `alpha_registry.db` where admission and
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

[admission]
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
- Add `[evo_daemon]`, `[admission]`, `[lifecycle_daemon]` to TOML and
  corresponding dataclasses to `config.py`.
- Run all 434 existing tests to confirm no regression.

#### Step 2: evo daemon (no trade downtime)

- Implement `daemon/evo.py`. Extract logic from `PipelineRunner._evolve()`.
- Add `evo-daemon` CLI subcommand.
- Add `alpha-os-evo@.service` systemd unit.
- Test: start evo daemon, verify candidates appear in DB.
- The existing `alpha-os.service` continues running unchanged.

#### Step 3: admission daemon (no trade downtime)

- Implement `daemon/admission.py`. Extract logic from
  `PipelineRunner._validate()` and `_adopt()`.
- Add `admission-daemon` CLI subcommand.
- Add `alpha-os-admission@.service` systemd unit.
- Test: verify candidates flow through validation to alphas table.

#### Step 4: trade cycle slimming (~1 min downtime)

- Add `skip_lifecycle` parameter to `Trader.run_cycle()`.
- Switch diversity source to `diversity_cache` table when
  `admission.enabled = true`.
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

The admission daemon's incremental approach computes correlation only between new
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

**alpha-os-admission@.service** (continuous poller):

```ini
[Unit]
Description=Alpha-OS admission daemon (%i)
After=network.target signal-noise-scheduler.service

[Service]
Type=simple
User=dev
WorkingDirectory=/home/dev/projects/alpha-os
ExecStart=/home/dev/projects/alpha-os/.venv/bin/python \
    -m alpha_os admission-daemon --asset %i
Restart=on-failure
RestartSec=120
MemoryHigh=500M
MemoryMax=700M
StandardOutput=append:/home/dev/projects/alpha-os/data/%i/logs/admission.log
StandardError=append:/home/dev/projects/alpha-os/data/%i/logs/admission.log
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
| alpha-os-admission | 500M | 700M | Admission + diversity computation |
| alpha-os-lifecycle | 300M | 500M | Daily oneshot, short-lived |
| OS + system services | ~500M | — | systemd, tailscale, sshd, etc. |
| **Total MemoryHigh** | **~3000M** | | Lifecycle and admission rarely overlap |

Effective peak usage is ~2.5GB. 8GB provides comfortable headroom with
no swap usage expected.

### Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| SQLite BUSY on alpha_registry.db | Admission and lifecycle write to `alphas` concurrently | WAL mode + `busy_timeout=30000ms`. Writes are millisecond-scale; 30s retry is more than sufficient. |
| evo daemon memory leak | OOM kill → candidates lost | systemd `MemoryMax=600M` hard cap. RSS monitoring + pop_size halving in daemon. `Restart=on-failure` for auto-recovery. Candidates already written to DB survive. |
| Phase 4 testnet disruption | Testnet validation streak resets | Steps 1-3 add no risk (new code only). Step 4 uses `enabled=false` default. Rollback: flip flag + restart (~1 min). |
| candidates table bloat | DB file growth | GC job deletes `adopted`/`rejected` rows older than 30 days. Scheduled inside admission daemon. |
| diversity cache staleness | Trade weights based on outdated diversity | `computed_at` timestamp tracked. Cache valid for `diversity_recompute_days` (63 days). DORMANT alphas excluded from combination regardless of cache state. |
| Process ordering | Trade runs before admission finishes first batch | Not a problem. Trade uses whatever ACTIVE alphas exist at cycle time. New alphas appear in the next cycle automatically. Exploration and exploitation are intentionally decoupled. |

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
