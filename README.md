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
  - lifecycle / deployment / sizing design
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

## Naming And Boundary Plan

The current direction is to keep the architecture idea, but make the names and
surface APIs much more explicit.

What should stay:

- the registry remains the research ledger
- deployed alphas remain the explicit runtime subset
- admission, lifecycle, and deployment stay separate concerns

What should change:

- names that blur lifecycle and runtime allocation
- config keys that hide which layer a limit applies to
- logs and reports that say `active` without saying whether they mean
  `registry active` or `deployed`

Current working plan:

1. keep the two-stage model (`registry` and `deployed_alphas`)
2. document the boundary everywhere before making more renames
3. prefer explicit names such as `registry active` and `deployed alphas`
4. tighten config names over time when the observation window ends

The project should not collapse the layers just because the current names are
awkward. The design goal is still to separate research churn from deployed
trading behavior.

### Naming Rules

Use these labels consistently in docs, logs, and reviews:

- say `registry active` when you mean lifecycle eligibility in `alphas`
- say `deployed alphas` when you mean the runtime subset in `deployed_alphas`
- say `deployment` for runtime allocation policy, not `universe`
- avoid plain `active` and plain `max_alphas` when the layer is ambiguous
- prefer names that encode the layer, even if they are longer

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

# Refresh deployed alphas from the registry
python -m alpha_os refresh-deployed-alphas --asset BTC

# Run a named replay experiment and persist the artifact
python -m alpha_os replay-experiment \
  --name "candidate-1.10" \
  --start 2025-09-01 \
  --end 2026-03-05 \
  --registry-mode admission \
  --deployment-mode refresh \
  --source candidates \
  --set lifecycle.candidate_quality_min=1.10

# Check testnet readiness status
python -m alpha_os testnet-readiness

# Show the current observation snapshot
python -m alpha_os runtime-status --asset BTC

# Queue the hand-crafted BTC baseline into the local admission queue
python -m alpha_os seed-handcrafted --asset BTC --alpha-set baseline

# Run a TOML-defined historical replay matrix
python -m alpha_os replay-matrix --manifest experiments/observation_window.toml --max-workers 2

# Compare deployed-alpha policy variants around the current best range
python -m alpha_os replay-matrix --manifest experiments/deployment_window.toml --max-workers 2
```

## Configuration

Edit `config/default.toml` or override via environment.

Key sections: `[api]` (signal-noise endpoint), `[generation]`, `[backtest]`, `[validation]` (OOS Sharpe, PBO gates), `[risk]` (drawdown stages), `[trading]` (initial capital), `[deployment]` (deployed alpha slots and replacement policy), `[execution]` (VPIN/spread/imbalance thresholds), `[testnet]`.

## Terminology Notes

Recent CLI cleanup now prefers standard terms:

- `trade` is the runtime trading command and defaults to Binance testnet.
  Real-money trading is only `trade --real`.
- Registry state names are `candidate`, `active`, `dormant`, and `rejected`.
  The admission registry is a research ledger, not the live trading set.
- `deployed alphas` means the deployed subset that `trade` actually reads.
  It is refreshed explicitly with `refresh-deployed-alphas`.
- `monitor` is the ongoing post-adoption monitoring loop for registry alphas.
- `paper --replay` is historical replay of the current runtime decision stack.
- `testnet-readiness` is the operational readiness check for testnet trading.
- `validate` remains statistical alpha validation via purged walk-forward CV.
- The CLI and TOML both use `admission` / `admission-daemon`.
- Logs and reports distinguish `registry active`, `shortlist candidates`,
  `deployed alphas`, `shortlist candidates`, `selected alphas`, and
  `signals evaluated`.

State naming note:

- `active` means `registry active`, not necessarily `currently traded`
- `currently traded` means `deployed in deployed_alphas`

This distinction is important. If logs or docs collapse those meanings, it
quickly becomes technical debt.

Near-term naming direction:

- keep `active` as the lifecycle state name for now
- prefer `registry active` in logs, docs, and discussions
- prefer `deployed alphas` for the live runtime subset
- treat ambiguous labels such as plain `active` or plain `max_alphas` as debt
  to remove gradually, not as the desired end state

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
  `n_registry_active`, `n_deployed_alphas`, `n_selected_alphas`
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
- verify that the configured deployed alpha count remains stable across refreshes
- compare new readiness reports against the registry DB and service memory
- improve documentation, logging, and operational checklists that do not change strategy behavior

Current server profile:

- `admission.max_active_alphas = 1000`
- `deployment.max_alphas = 120`

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

## System Map

Read the current runtime through these four flows:

1. `registry -> deployed_alphas -> trade`
   - research candidates live in the registry
   - deployed alphas live in `deployed_alphas`
   - `trade` should only act on the deployed subset
2. `signal -> target -> intent -> executable order`
   - prediction, portfolio construction, execution planning, venue constraints
3. `admission -> lifecycle`
   - `admission` decides what enters `active`
   - `lifecycle` decides what leaves `active`
4. `replay -> testnet readiness`
   - `replay` compares logic on history
   - `testnet-readiness` measures runtime behavior that replay cannot fully capture

Recommended first files:

- [registry.py](/home/dev/projects/alpha-os/src/alpha_os/alpha/registry.py)
- [deployed_alphas.py](/home/dev/projects/alpha-os/src/alpha_os/alpha/deployed_alphas.py)
- [trader.py](/home/dev/projects/alpha-os/src/alpha_os/paper/trader.py)
- [planning.py](/home/dev/projects/alpha-os/src/alpha_os/execution/planning.py)
- [constraints.py](/home/dev/projects/alpha-os/src/alpha_os/execution/constraints.py)
- [admission.py](/home/dev/projects/alpha-os/src/alpha_os/daemon/admission.py)
- [lifecycle.py](/home/dev/projects/alpha-os/src/alpha_os/alpha/lifecycle.py)
- [replay.py](/home/dev/projects/alpha-os/src/alpha_os/experiments/replay.py)
- [testnet.py](/home/dev/projects/alpha-os/src/alpha_os/validation/testnet.py)

## Research Notes

- `alpha-os` behaves more like a participant-ranking system than a single model.
  A large candidate set exists, but only a small deployed subset should drive trading.
- If the simplified runtime still fails after the observation window, alpha
  diversity is a strong next bottleneck candidate.
- A separate bottleneck is experiment throughput: many useful comparisons still
  require source edits or serial runs, which slows learning.
- A useful next benchmark is a small hand-crafted BTC alpha set spanning trend,
  reversal, cross-asset, macro regime, sentiment, on-chain, derivatives, and
  microstructure ideas.
- The current runtime is still primarily directional. Long-term, the system
  should move toward distribution-aware forecasts instead of scalar conviction alone.

What better experiment throughput should look like:

- more experiments driven by config / manifests instead of source edits
- isolated temp registry / deployed-alpha set / output paths by default
- parallel historical replay as a first-class workflow
- continued serial handling for live-ish testnet runs that share one venue/account

See [DESIGN.md](/home/dev/projects/alpha-os/DESIGN.md) for the detailed design
discussion, including:

- runtime layer boundaries
- participant-system mapping
- diversity as a bottleneck candidate
- experiment harness as a bottleneck candidate
- hand-crafted BTC baseline candidates
- directional vs distributional forecasting
- sizing rationale and known limitations

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
- `--deployment-mode current` uses the currently deployed alpha set if present.
- `--deployment-mode refresh` refreshes a temporary deployed alpha set inside the experiment before replay.
- Use repeated `--set path=value` flags for temporary config overrides.

## Design

See [DESIGN.md](DESIGN.md) for detailed architecture, alpha lifecycle, and DSL specification.
See [ROADMAP.md](ROADMAP.md) for the multi-timeframe evolution roadmap (Phase 1-3 complete, Phase 4-5 planned).
