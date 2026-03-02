# alpha-os

Autonomous alpha generation and trading system for crypto (BTC/USDT).
3-layer architecture: strategic (daily), tactical (hourly), execution (minute).
Generates trading signals using genetic programming (GP + MAP-Elites),
validates them with walk-forward cross-validation, and executes via Binance.

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
├── paper/        Paper trading simulator + tracker
├── execution/    Trade executors (Paper, Binance), ExecutionOptimizer
├── trading/      EventDrivenTrader, TacticalTrader
├── risk/         Position sizing, drawdown stages, circuit breaker
├── data/         DataStore (signal-noise integration, subdaily resolution support)
├── forward/      Forward test runner
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
python -m alpha_os validate --alpha-id <id>

# Forward-test adopted alphas
python -m alpha_os forward

# Paper trade
python -m alpha_os paper --once

# Live trade (testnet)
python -m alpha_os live --once --testnet --asset BTC

# Event-driven mode (WebSocket, auto-triggers on market events)
python -m alpha_os live --event-driven --testnet --asset BTC

# Layer 2 tactical evolution
python -m alpha_os evolve --layer L2 --generations 30

# Check testnet validation status
python -m alpha_os validate-testnet
```

## Configuration

Edit `config/default.toml` or override via environment.

Key sections: `[api]` (signal-noise endpoint), `[generation]`, `[backtest]`, `[validation]` (OOS Sharpe, PBO gates), `[risk]` (drawdown stages), `[trading]` (initial capital), `[execution]` (VPIN/spread/imbalance thresholds), `[testnet]`.

## Testing

```bash
pytest tests/
ruff check src/
```

## Design

See [DESIGN.md](DESIGN.md) for detailed architecture, alpha lifecycle, and DSL specification.
See [ROADMAP.md](ROADMAP.md) for the multi-timeframe evolution roadmap (Phase 1-3 complete, Phase 4-5 planned).
