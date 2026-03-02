# alpha-os Agent Guide

## Project

Autonomous alpha generation + BTC/USDT trading system.
Python 3.12, S-expression DSL, GP + MAP-Elites evolution, Binance execution.

3-Layer architecture:
- **Layer 3 (Strategic)**: Daily signals → direction bias (GP evolution on daily DSL)
- **Layer 2 (Tactical)**: Hourly signals → entry/exit timing (TacticalTrader, 17 hourly features)
- **Layer 1 (Execution)**: Minute signals → optimal execution (ExecutionOptimizer, VPIN/spread/imbalance)

## Structure

```
src/alpha_os/       Main package
  data/             DataStore, universe (daily + hourly + microstructure features)
  dsl/              S-expression DSL parser, evaluator, GP generator
  execution/        Executor ABC, BinanceExecutor, ExecutionOptimizer
  risk/             Position sizing, circuit breaker
  trading/          PaperTrader, EventDrivenTrader, TacticalTrader
config/             TOML configuration
scripts/            Operational scripts (cron, e2e tests)
tests/              pytest test suite (423 tests)
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
- **SQLite** for persistence (alpha_cache.db, alpha_registry.db, paper_trading.db)
- **S-expression DSL** for alpha expressions: `(neg f1)`, `(ts_mean f2 10)`, `(if_gt f1 f2 f3 f4)`
- **Executor interface** (`Executor` ABC in `execution/executor.py`) — Paper and Binance implementations
- **ExecutionOptimizer** — microstructure-aware execution timing (optional, plugs into BinanceExecutor)
- **Config** loaded from `config/default.toml` via `Config.load()` (includes `[execution]` thresholds)

## Testing

- pytest with fixtures in `tests/conftest.py`
- `synthetic_data()` — 300 days of reproducible price data (3 features)
- `registry()` / `populated_registry()` — temp DB backed alpha registries
- `data_store()` — DataStore pre-populated with synthetic signals

## Data Files (gitignored)

- `data/alpha_cache.db` — alpha evaluation cache
- `data/alpha_registry.db` — adopted alphas registry
- `data/forward_returns.db` — forward return data
- `data/paper_trading.db` — paper trading history
- `data/metrics/` — testnet validation state and reports
- `data/logs/` — daily live trading logs

## Secrets

- `~/.secrets/binance` — Binance testnet API keys
- `~/.secrets/binance_real` — Binance production API keys
- Format: `export BINANCE_API_KEY=...` / `export BINANCE_SECRET_KEY=...`
- Never commit secrets. Loaded by `_load_secrets()` in `execution/binance.py`

## signal-noise Integration

- REST API at `http://127.0.0.1:8000` (same server)
- WebSocket at `ws://127.0.0.1:8000/ws/signals` for real-time events
- `DataStore.sync()` fetches incremental data via `SignalClient` (supports `resolution` param)
- `alpha_cache.db` caches locally (works offline if API is down)
- EventDrivenTrader subscribes to signal events via WebSocket
- ExecutionOptimizer reads realtime microstructure signals via `SignalClient.get_latest()`

## Deployment

Server-specific deployment info is in `AGENTS.override.md` (not committed).
