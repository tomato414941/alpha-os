# alpha-os Agent Guide

## Project

Autonomous alpha generation + trading system.
Python 3.12, S-expression DSL, pure MAP-Elites evolution, multi-venue execution.

Asset classes: crypto (BTC/ETH/SOL), US stocks, ETFs, prediction markets (Polymarket).
Venues: Binance (crypto), Alpaca (equities), Polymarket (prediction markets), Paper.

Current trusted runtime:
- `hypothesis-seeder`
- `sync-signal-cache`
- `produce-predictions`
- `trade --once --venue paper`
- `runtime-status`

Legacy tactical, replay, and registry-era paths still exist in places, but are
not the current mainline.

## Structure

```
src/alpha_os/       Main package
  data/             DataStore, universe (daily + hourly + microstructure features)
  dsl/              S-expression DSL parser, evaluator, GP generator
  evolution/        Pure MAP-Elites discovery pool (4D behavioral grid)
  execution/        Executor ABC, BinanceExecutor, AlpacaExecutor, PolymarketExecutor, ExecutionOptimizer
  risk/             Position sizing, circuit breaker, BinaryOutcomeRiskManager (Kelly)
  paper/            Single-asset runtime, legacy replay helpers
config/             TOML configuration
scripts/            Operational scripts (cron, e2e tests)
tests/              pytest test suite
data/               Runtime data (SQLite DBs, logs) — gitignored
```

## Development

```bash
source .venv/bin/activate
pip install -e ".[dev]"
pip install -e ~/projects/signal-noise   # local dependency

# Run tests
pytest tests/

# Lint
ruff check src/

# CLI
python -m alpha_os --help
```

## Conventions

- **Dataclasses** for all data structures (Fill, Order, AlphaRecord, ExecutionConfig, etc.)
- **SQLite** for persistence (`signal_cache.db`, `hypotheses.db`, `paper_trading.db`)
- **S-expression DSL** for alpha expressions: `(neg f1)`, `(ts_mean f2 10)`, `(if_gt f1 f2 f3 f4)`
- **Executor interface** (`Executor` ABC in `execution/executor.py`) — Paper, Binance, Alpaca, Polymarket implementations
- **Venue auto-detection** — `infer_venue(asset)` maps asset → venue; CLI `--venue` for override
- **ExecutionOptimizer** — microstructure-aware execution timing (optional, plugs into BinanceExecutor)
- **Secrets** — shared `execution/secrets.py` (`load_secrets`, `get_env_or_secret`); per-venue files in `~/.secrets/`
- **Config** loaded from `config/default.toml` via `Config.load()` (includes `[execution]`, `[alpaca]`, `[polymarket]`)
- **MAP-Elites** discovery pool: 4D behavioral grid (persistence × activity × price_beta × vol_sensitivity), 8×8×8×8 = 4,096 cells
- **TC (True Contribution)** weighting: leave-one-out ensemble Sharpe for portfolio weights + lifecycle demotion (TC ≤ 0 → dormant)

## Terminology Policy

- Prefer standard industry terminology in new docs, config keys, commands, and logs.
- Keep legacy project-specific names only when needed for backward compatibility.
- Distinguish statistical alpha validation (`validate`) from operational validation
  (`admission-daemon`, `testnet-readiness`).
- Distinguish Binance `testnet` from real-money `trade --real`.
- Distinguish registry state counts from the per-cycle trading subset in logs and reports.

## Testing

- pytest with fixtures in `tests/conftest.py`
- `synthetic_data()` — 300 days of reproducible price data (3 features)
- `registry()` / `populated_registry()` — temp DB backed alpha registries
- `data_store()` — DataStore pre-populated with synthetic signals

## Data Files (gitignored)

- `data/signal_cache.db` — local signal data cache
- `data/hypotheses.db` — canonical hypothesis store
- `data/alpha_registry.db` — legacy registry substrate; not the current runtime source of truth
- `data/forward_returns.db` — forward return data
- `data/paper_trading.db` — paper trading history
- `data/metrics/` — testnet readiness state and reports
- `data/logs/` — daily trade runtime logs

## Secrets

All venues use `execution/secrets.py` (`load_secrets`, `get_env_or_secret`).

- `~/.secrets/binance` — Binance testnet API keys (`BINANCE_API_KEY`, `BINANCE_SECRET_KEY`)
- `~/.secrets/binance_real` — Binance production API keys
- `~/.secrets/alpaca` — Alpaca paper trading keys (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`)
- `~/.secrets/alpaca_real` — Alpaca production keys
- `~/.secrets/polymarket` — Polymarket keys (`POLYMARKET_PRIVATE_KEY`, `POLYMARKET_API_KEY`)
- Format: `export KEY=value` or `KEY=value`
- Never commit secrets. Env vars take precedence over files.

## signal-noise Integration

- REST API at `http://127.0.0.1:8000` (same server)
- WebSocket at `ws://127.0.0.1:8000/ws/signals` for real-time events
- `DataStore.sync()` fetches incremental data via `SignalClient` (supports `resolution` param)
- `signal_cache.db` caches signals locally (works offline if API is down)
- EventDrivenTrader subscribes to signal events via WebSocket
- ExecutionOptimizer reads realtime microstructure signals via `SignalClient.get_latest()`

## Deployment

Server-specific deployment info is in `AGENTS.override.md` (not committed).
