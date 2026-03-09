# alpha-os

Autonomous alpha generation and trading system for any asset class — crypto, equities, commodities, derivatives, prediction markets, and beyond.
3-layer architecture: strategic (daily), tactical (hourly), execution (minute).
Generates trading signals using genetic programming (GP + MAP-Elites),
validates them with walk-forward cross-validation, and executes via pluggable executors (Binance, Alpaca, etc.).

## Document Guide

Use the docs in this order:

- `README.md`
  - current project entry point
  - runtime concepts
  - current operating posture
  - where to look next
- `DESIGN.md`
  - architectural decisions
  - lifecycle / universe / sizing design
  - known limitations and long-term direction
- `AGENTS.override.md`
  - server operations
  - deploy / restart / verification steps
  - current testnet observation workflow
- `ROADMAP.md`
  - larger planned expansion work

If a topic appears in multiple files, prefer:

- `README.md` for current practical guidance
- `DESIGN.md` for deeper rationale
- `AGENTS.override.md` for concrete commands

## Current Focus

The project is currently in a short observation phase after recent runtime
simplification. That means:

- runtime behavior should remain mostly stable for `24 hours` or `3-5` cycles
- docs, observability, and system understanding are in scope
- large runtime changes are out of scope until the observation window ends

## Architecture

```
Layer 3: Strategic (Daily)    — Direction bias via GP evolution on 448+ daily signals
Layer 2: Tactical (Hourly)    — Entry/exit timing via 17 hourly features (funding, OI, liquidations)
Layer 1: Execution (Minute)   — VPIN/spread/imbalance-based optimal execution timing

src/alpha_os/
├── dsl/          S-expression DSL (parser, evaluator, operators, GP templates)
├── evolution/    GP + MAP-Elites alpha evolution
├── backtest/     Backtest engine, cost model, metrics
├── validation/   Purged Walk-Forward CV, PBO, DSR, FDR
├── alpha/        Alpha evaluator, registry, lifecycle, combiner
├── paper/        Paper trading simulator + tracker, EventDrivenTrader, TacticalTrader
├── execution/    Trade executors (Paper, Binance), ExecutionOptimizer
├── risk/         Position sizing, drawdown stages, circuit breaker
├── data/         DataStore (signal-noise integration, subdaily resolution support)
├── forward/      Ongoing alpha monitoring (legacy package name)
├── governance/   Adoption gates, audit log
└── pipeline/     Scheduler, pipeline runner
```

Data source: [signal-noise](https://github.com/tomato414941/signal-noise) — time series collector providing 1,307+ signals via REST API + WebSocket.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# signal-noise (same server, required for data)
pip install -e ~/projects/signal-noise
```

## CLI

```bash
# Generate alpha expressions
python -m alpha_os generate --count 10000

# Backtest generated alphas
python -m alpha_os backtest

# Evolve alphas via GP + MAP-Elites
python -m alpha_os evolve --generations 30

# Validate an alpha with purged walk-forward CV
python -m alpha_os validate --expr '<dsl-expression>' --asset BTC

# Monitor adopted alphas
python -m alpha_os monitor

# Paper trade
python -m alpha_os paper --once

# Historical replay
python -m alpha_os paper --replay --start 2025-09-01 --end 2026-03-05

# Trade command (Binance testnet by default; add --real for actual capital)
python -m alpha_os trade --once --asset BTC

# Event-driven mode (WebSocket, auto-triggers on market events)
python -m alpha_os trade --event-driven --asset BTC

# Layer 2 tactical evolution
python -m alpha_os evolve --layer 2 --generations 30

# Run the candidate admission daemon
python -m alpha_os admission-daemon --asset BTC

# Rebuild registry states from validated candidates
python -m alpha_os rebuild-registry --asset BTC --source candidates

# Refresh the deployed trading universe from the registry
python -m alpha_os refresh-universe --asset BTC

# Run a named replay experiment and persist the artifact
python -m alpha_os replay-experiment \
  --name "candidate-1.10" \
  --start 2025-09-01 \
  --end 2026-03-05 \
  --registry-mode admission \
  --universe-mode refresh \
  --source candidates \
  --set lifecycle.candidate_quality_min=1.10

# Check testnet readiness status
python -m alpha_os testnet-readiness

# Show the current observation snapshot
python -m alpha_os runtime-status --asset BTC
```

## Configuration

Edit `config/default.toml` or override via environment.

Key sections: `[api]` (signal-noise endpoint), `[generation]`, `[backtest]`, `[validation]` (OOS Sharpe, PBO gates), `[risk]` (drawdown stages), `[trading]` (initial capital), `[universe]` (deployed alpha slots and replacement policy), `[execution]` (VPIN/spread/imbalance thresholds), `[testnet]`.

## Terminology Notes

Recent CLI cleanup now prefers standard terms:

- `trade` is the runtime trading command and defaults to Binance testnet.
  Real-money trading is only `trade --real`.
- Registry state names are `candidate`, `active`, `dormant`, and `rejected`.
  The admission registry is a research ledger, not the live trading set.
- `trading universe` means the deployed subset that `trade` actually reads.
  It is refreshed explicitly with `refresh-universe`.
- `monitor` is the ongoing post-adoption monitoring loop for registry alphas.
- `paper --replay` is historical replay of the current runtime decision stack.
- `testnet-readiness` is the operational readiness check for testnet trading.
- `validate` remains statistical alpha validation via purged walk-forward CV.
- The CLI and TOML both use `admission` / `admission-daemon`.
- Logs and reports distinguish `registry active`, `shortlist candidates`,
  `universe deployed`, `shortlist candidates`, `selected alphas`, and
  `signals evaluated`.

State naming note:

- `active` means `registry active`, not necessarily `currently traded`
- `currently traded` means `deployed in trading_universe`

This distinction is important. If logs or docs collapse those meanings, it
quickly becomes technical debt.

## Registry Control

The research registry is allowed to change continuously, but it should not
grow without bound.

- `admission.max_active_alphas` is the hard cap for `alphas.state=active`.
- `0` disables the cap. Any positive value enables it.
- When the cap is reached, a weaker incoming alpha is rejected.
- When the cap is reached and the incoming alpha is stronger, the weakest
  active alpha is demoted to `dormant` first.
- If the registry is already above the cap, admission first prunes the
  weakest active rows down to the limit, then evaluates the new alpha.

This is intentionally simple. It avoids adding a new pool or a second
deployment lifecycle just to control registry growth.

## Runtime Observation

For current testnet operation, watch these values first:

- registry DB counts: `active`, `dormant`, `rejected`
- readiness report fields:
  `n_registry_active`, `n_universe_deployed`, `n_selected_alphas`
- trading outcomes:
  `n_fills`, `n_skipped_deadband`, `n_order_failures`, `daily_pnl`
- service health:
  `alpha-os.service` and `alpha-os-admission@BTC.service` memory usage

The readiness files have different roles:

- `data/BTC/metrics/testnet_readiness.json`
  aggregated readiness state such as consecutive success days
- `data/BTC/metrics/testnet_readiness_reports.jsonl`
  per-cycle reports with fills, skips, reconciliation, and registry counts

The registry DB can change before the next scheduled trade cycle writes a new
readiness report, so DB counts and the latest report may briefly disagree.

### Observation Window

Do not pause development for multiple days just to watch this strategy.

- use a short observation window: about `24 hours` or `3-5` scheduled trade cycles
- avoid large runtime changes during that window so the result stays attributable
- after that window, make a go / no-go call instead of extending observation by default

Typical no-go signals:

- fills remain near zero
- `n_skipped_deadband` dominates and the runtime rarely trades
- daily PnL shows no improvement signal
- `n_registry_active` or service memory becomes unstable again

## Current Operating Posture

After the recent runtime cleanup, the project is in a short observation phase.

### What We Can Do Now

- observe `trade` cycles and `admission` stability without changing runtime behavior
- verify that `admission.max_active_alphas` keeps the registry bounded
- verify that `trading_universe=30` remains stable across refreshes
- compare new readiness reports against the registry DB and service memory
- improve documentation, logging, and operational checklists that do not change strategy behavior

### What We Cannot Conclude Yet

- that the strategy has a durable edge
- that the current testnet profile is strong enough for real capital
- that low fill counts are a feature rather than a sign of insufficient opportunity
- that replay improvements are large enough to matter economically

These require the short observation window to complete first.

### What We Should Not Do During This Window

- add new runtime states, pools, or deployment paths
- do large refactors that change trade behavior again
- run broad parameter sweeps that invalidate attribution
- treat the current profile as production-ready before the observation window ends

### Next Decision

After `24 hours` or `3-5` scheduled trade cycles, make a go / no-go call.

- `go`: keep the simplified runtime and continue with the next bottleneck
- `no-go`: stop extending this configuration and move to the next strategy hypothesis

## Runtime Boundaries

The runtime is being standardized around five layers:

- `Prediction`: produce alpha and combined signals
- `Portfolio Construction`: decide target holdings and risk-adjusted exposure
- `Execution Planning`: convert target gaps into order intents
- `Venue Constraints`: enforce min notional, precision, lot size, and fees
- `Execution`: submit, retry, and reconcile venue-valid orders

This boundary exists to keep venue-specific constraints from silently shaping
strategy behavior. Executors should receive only orders that are already valid
for the target venue.

## Runtime Cleanup Direction

The current runtime is more modular than before, but some policy layers are
still heavier than they should be.

- Keep: `trading_universe`, explicit execution handoffs, shared cost model.
- Reduce: optimizer hard blocks, stale circuit-breaker carryover, and policy branching inside `trade`.

The next cleanup goal is not another large rewrite. It is to preserve the
runtime boundaries while removing decision paths that mostly add delay,
fallbacks, and operational ambiguity.

## How To Read The System

If you are trying to understand the project during an observation window,
read it in this order:

1. `registry -> trading_universe -> trade`
   - Research alphas live in the registry.
   - Deployed alphas live in `trading_universe`.
   - `trade` should only act on the deployed subset.
   - Start with
     [registry.py](/home/dev/projects/alpha-os/src/alpha_os/alpha/registry.py),
     [trading_universe.py](/home/dev/projects/alpha-os/src/alpha_os/alpha/trading_universe.py),
     [trader.py](/home/dev/projects/alpha-os/src/alpha_os/paper/trader.py)

2. `signal -> target -> intent -> executable order`
   - This is the current runtime boundary.
   - Start with
     [planning.py](/home/dev/projects/alpha-os/src/alpha_os/execution/planning.py),
     [constraints.py](/home/dev/projects/alpha-os/src/alpha_os/execution/constraints.py),
     [binance.py](/home/dev/projects/alpha-os/src/alpha_os/execution/binance.py)

3. `admission -> lifecycle`
   - `admission` decides what enters `active`.
   - `lifecycle` decides what leaves `active`.
   - Start with
     [admission.py](/home/dev/projects/alpha-os/src/alpha_os/daemon/admission.py),
     [lifecycle.py](/home/dev/projects/alpha-os/src/alpha_os/alpha/lifecycle.py)

4. `replay -> testnet readiness`
   - `replay` compares logic on historical data.
   - `testnet-readiness` measures runtime behavior that replay cannot fully capture.
   - Start with
     [replay.py](/home/dev/projects/alpha-os/src/alpha_os/experiments/replay.py),
     [testnet.py](/home/dev/projects/alpha-os/src/alpha_os/validation/testnet.py)

This order helps separate research, deployment, runtime planning, and
operational verification.

## Participant-System Mapping

It can be useful to think about `alpha-os` like a participant-ranking system
such as Numerai or a market with many forecasters. The analogy is not exact,
but the governance mapping is useful:

| Participant system idea | `alpha-os` counterpart |
| ----------------------- | ---------------------- |
| participant / forecaster | one alpha expression |
| submission set | registry (`alphas`) |
| eligible participant | `state=active` in the registry |
| live allocation / stake | deployed `trading_universe` slot |
| promotion | enter or remain in `trading_universe` |
| demotion | move to `dormant` or fail to stay deployed |
| payout proxy | blended quality, deployment score, and realized cycle contribution |
| uniqueness / originality | diversity and correlation filtering |

The important takeaway is that `alpha-os` is not trying to trade every
participant. It is trying to govern a large candidate set and deploy only a
small subset at any given time.

### Diversity Note

If the current simplified runtime still fails to show a useful edge after the
observation window, alpha diversity is one of the strongest next bottleneck
candidates.

The target is not "more alphas" by itself. The target is:

- more independent alphas
- less duplication across close variants
- better selection and allocation across distinct views

That is the direction in which `alpha-os` would become more similar to a
participant system such as Numerai or a forecasting market with many
independent contributors.

## Hand-Crafted BTC Baseline Set

A useful next baseline is a small set of manually designed BTC alphas. The
goal is not to beat the full search space immediately. The goal is to create a
clear, explainable benchmark that the generated/evolved pool should be able to
beat.

These are good first candidates because they represent distinct ideas rather
than many close variants:

1. Momentum
   - `(roc_20 btc_ohlcv)`
2. Short-horizon mean reversion
   - `(neg (zscore (roc_5 btc_ohlcv)))`
3. BTC vs equity relative strength
   - `(sub (roc_20 btc_ohlcv) (roc_20 sp500))`
4. Volatility-conditioned trend
   - `(if_gt vix_close 25.0 (neg (roc_10 btc_ohlcv)) (roc_10 btc_ohlcv))`
5. Sentiment-conditioned reversal
   - `(if_gt fear_greed 70.0 (neg btc_ohlcv) btc_ohlcv)`
6. On-chain activity imbalance
   - `(sub (zscore btc_mempool_size) (zscore btc_hashrate))`
7. Funding-rate mean reversion
   - `(sub funding_rate_btc (mean_5 funding_rate_btc))`
8. Order-book imbalance fade
   - `(neg book_imbalance_btc)`
9. Spread compression preference
   - `(neg spread_bps_btc)`
10. VPIN-conditioned flow signal
   - `(if_gt vpin_btc 0.8 (neg trade_flow_btc) trade_flow_btc)`

Why this set:

- it spans trend, reversal, cross-asset, macro regime, sentiment, on-chain,
  derivatives, and microstructure ideas
- each alpha is easy to explain and debug
- it can act as a benchmark against the evolved candidate pool

These expressions are intended as a research baseline first. They should go
through the same validation, admission, and deployment path as generated
alphas instead of being special-cased into live trading.

### Position sizing

```
direction  = sign(weighted_signal_mean)
size       = consensus × dd_scale
position   = direction × clip(size) × max_position_pct × portfolio_value
```

- **Signal consensus**: measures alpha agreement — `|mean| / (|mean| + std)`. Unanimous → full conviction; split → reduced position.
- **Drawdown scaling**: position shrinks as portfolio drawdown deepens (3 stages).

### Distributional Direction

The current runtime is still primarily directional:

- each alpha emits a signed scalar signal
- the combined runtime signal represents direction plus conviction
- position sizing is based on that combined scalar, not on a full predictive distribution

This is acceptable for the current simplified runtime, but it is not the
intended long-term endpoint.

The longer-term direction is to move toward distribution-aware forecasts:

- expected return plus uncertainty first
- then distribution-aware sizing and selection
- eventually quantiles / tail-aware forecasts instead of scalar-only conviction

In other words: the current system predicts direction better than it predicts
the full future return distribution, and that is a known design limitation.

## Testing

```bash
pytest tests/
ruff check src/
```

## Experiment Workflow

- Use `replay-experiment` for named historical experiments instead of ad-hoc shell notes.
- Artifacts are written to `data/<ASSET>/experiments/`.
- Each run appends a summary row to `index.jsonl` and stores the full payload in a timestamped `.json`.
- `--registry-mode current` replays the current registry as-is.
- `--registry-mode admission` rebuilds a temporary registry from `alphas` or `candidates` first, which is better for gate experiments.
- `--universe-mode current` uses the currently deployed universe if present.
- `--universe-mode refresh` refreshes a temporary deployed universe inside the experiment before replay.
- Use repeated `--set path=value` flags for temporary config overrides.

## Design

See [DESIGN.md](DESIGN.md) for detailed architecture, alpha lifecycle, and DSL specification.
See [ROADMAP.md](ROADMAP.md) for the multi-timeframe evolution roadmap (Phase 1-3 complete, Phase 4-5 planned).
