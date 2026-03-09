# alpha-os

Autonomous alpha generation and trading system for any asset class — crypto, equities, commodities, derivatives, prediction markets, and beyond.
3-layer architecture: strategic (daily), tactical (hourly), execution (minute).
Generates trading signals using genetic programming (GP + MAP-Elites),
validates them with walk-forward cross-validation, and executes via pluggable executors (Binance, Alpaca, etc.).

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

### Position sizing

```
direction  = sign(weighted_signal_mean)
size       = consensus × dd_scale
position   = direction × clip(size) × max_position_pct × portfolio_value
```

- **Signal consensus**: measures alpha agreement — `|mean| / (|mean| + std)`. Unanimous → full conviction; split → reduced position.
- **Drawdown scaling**: position shrinks as portfolio drawdown deepens (3 stages).

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
