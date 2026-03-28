# alpha-os

Hypotheses-first trading runtime under recovery.

Current trusted path:

- `hypothesis-seeder`
- `sync-signal-cache`
- `produce-predictions`
- `trade --once --venue paper`
- `runtime-status`
- `run-sleeves-once`
- `compare-sleeves`

The active recovery goal is not to restore every legacy subsystem. It is to
make the bounded hypotheses-first runtime trustworthy again and only then
decide which legacy research paths deserve a rewrite.

Command-surface rule:

- current runtime commands live at the root
- research-only commands live under `research ...`
- legacy / archive commands live under `legacy ...`

## Document Guide

Prefer this order:

1. `README.md`
   - project entry point
   - current runtime path
2. `OPERATING_BOUNDARIES.md`
   - current operating truth
   - trusted / untrusted boundary
3. `docs/README.md`
   - document index
   - which design note to read next

After that:

- `docs/portfolio-runtime-principles.md`
  - current portfolio / allocation semantics
- `DESIGN.md`
  - greenfield and long-horizon architecture
- `AGENTS.override.md`
  - machine-local operations

Exploratory or archival files are intentionally lower priority:

- `docs/exploratory/ROADMAP.md`
- `docs/exploratory/PREDICTION_TARGETS.md`
- `docs/exploratory/TRADING_UNIVERSE_DESIGN.md`

## Current Runtime Path

The current runtime is bounded, multi-sleeve, and hypotheses-first.
Today that means:

- `BTC` as the bounded reference sleeve
- `ETH` as the first non-reference validation sleeve
- repeated bounded oneshot runs instead of always-on services

```
signal-noise
    │
    ▼
hypothesis-seeder  ->  hypotheses.db
    │                     │
    └──── sync-signal-cache / produce-predictions
                          │
                          ▼
              trade --once --venue paper
                          │
                          ▼
          runtime-status / compare-sleeves / readiness
```

What is current:

- canonical runtime store: `data/hypotheses.db`
- canonical local market-data cache: `data/signal_cache.db`
- canonical bounded trade entrypoint: `trade --once --venue paper`
- readiness is accumulated from repeated bounded runs

What is no longer current:

- `cross-trade`
- `submit`
- `refresh-deployed-alphas`
- `rebuild-managed-alphas`
- `seed-handcrafted`
- `analyze-diversity`

Those paths were tied to the legacy managed/deployed registry workflow and are
either removed or treated as legacy experimental material.

### Current Status

- **operating posture**: `unsafe to run`
- **trusted bounded path**: hypotheses-first `paper` runtime only
- **current source of truth**: see `OPERATING_BOUNDARIES.md` for runtime boundaries and restart gate
- **scheduler posture**: do not re-enable old units; use the recovery draft only as a template

### Simplified Direction

The near-term simplification direction is:

- one canonical table: `hypotheses`
- one canonical target: `forward_residual_return`
- one canonical horizon: `20D2L`
- one canonical universe: `core_universe_1000`
- no separate `candidates` or `deployed_alphas` in the target design

This is a deliberate temporary simplification to restore trust in the system.
See `OPERATING_BOUNDARIES.md` for the current migration target.

### Legacy Experimental Paths

These remain in the repo, but they are not the current runtime:

- `research paper-replay`
- `research replay-experiment`
- `research replay-matrix`
- `legacy admission-daemon`
- `legacy prune-stale-candidates`
- `managed_alphas` / `deployed_alphas`

Treat them as legacy experimental or archival material unless they are
explicitly rewritten around hypotheses-first inputs.

## Architecture Notes

For deeper rationale and longer-term design, prefer:

- [DESIGN.md](/home/dev/projects/alpha-os/DESIGN.md)
- [OPERATING_BOUNDARIES.md](/home/dev/projects/alpha-os/OPERATING_BOUNDARIES.md)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# signal-noise client dependency only
pip install signal-noise
```

## CLI

```bash
# Seed or refresh hypotheses
python -m alpha_os_recovery hypothesis-seeder --config config/dev.toml

# Sync only the signals required by live hypotheses
python -m alpha_os_recovery sync-signal-cache --asset BTC --from-hypotheses --strict --config config/dev.toml

# Produce bounded hypothesis predictions
python -m alpha_os_recovery produce-predictions --asset BTC --strict --config config/dev.toml

# Run one bounded paper cycle
python -m alpha_os_recovery trade --once --asset BTC --venue paper --strict --config config/dev.toml

# Check readiness / runtime observation
python -m alpha_os_recovery runtime-status --asset BTC --config config/dev.toml

# Run the bounded multi-sleeve loop
python -m alpha_os_recovery run-sleeves-once --assets BTC,ETH --config config/dev.toml --score-limit 12
python -m alpha_os_recovery compare-sleeves --assets BTC,ETH --config config/dev.toml

# Legacy experimental paths
python -m alpha_os_recovery research paper-replay --start 2025-09-01 --end 2026-03-05
python -m alpha_os_recovery research replay-experiment --name smoke --start 2025-09-01 --end 2026-03-05
python -m alpha_os_recovery legacy admission-daemon --asset BTC --config config/dev.toml
```

## alpha_os

The new `alpha_os` package is a separate bounded evaluation engine.
It is intentionally narrower than `alpha_os_recovery`.

Current scope:

- `1 asset`: `BTC`
- `paper-only`
- `1 target`: `residual_return_1d`
- primary bounded path:
  `register-hypothesis -> record-prediction -> finalize-observation -> update-state`
- convenience wrapper:
  `apply-cycle`

Primary bounded path:

```bash
python -m alpha_os register-hypothesis --db data/v1/runtime.db --hypothesis-id hyp_momo
python -m alpha_os record-prediction --db data/v1/runtime.db --date 2026-03-27 --hypothesis-id hyp_momo --prediction 0.05
python -m alpha_os finalize-observation --db data/v1/runtime.db --date 2026-03-27 --observation -0.02
python -m alpha_os update-state --db data/v1/runtime.db --date 2026-03-27 --hypothesis-id hyp_momo
```

Convenience wrapper path:

```bash
# Generate one or more deterministic evaluation inputs from signal-noise BTC daily closes
python -m alpha_os generate-cycle-input --date 2026-03-27 --hypothesis-id hyp_momo --out data/v1/cycle.json
python -m alpha_os generate-cycle-inputs --start-date 2026-03-27 --end-date 2026-03-31 --hypothesis-id hyp_momo --out data/v1/cycles.json

# Apply one evaluation or a deterministic backfill range through the wrapper path
python -m alpha_os register-hypothesis --db data/v1/runtime.db --hypothesis-id hyp_momo
python -m alpha_os apply-cycle --db data/v1/runtime.db --input data/v1/cycle.json
python -m alpha_os apply-backfill --db data/v1/runtime.db --start-date 2026-03-27 --end-date 2026-03-31 --hypothesis-id hyp_momo --out data/v1/cycles.json

# Inspect aggregate state and per-evaluation provenance
python -m alpha_os status --db data/v1/runtime.db
python -m alpha_os show-evaluations --db data/v1/runtime.db --limit 10
```

## Configuration

Edit `config/default.toml` or override via environment.

Key sections: `[api]`, `[generation]`, `[backtest]`, `[validation]`, `[risk]`, `[trading]`, `[execution]`, `[testnet]`.

If the remote `signal-noise` endpoint is protected, set `ALPHA_OS_SIGNAL_NOISE_API_KEY`
in the runtime environment. `alpha-os` will automatically pass it to
`signal-noise` client calls without storing the secret in repo config.

## Terminology Notes

- prefer `hypothesis` for a predictive unit
- reserve `alpha` for excess-return outcome or legacy code names
- `live hypotheses` is the runtime subset
- `research paper-replay` and `research replay-experiment` are legacy experimental paths
- `trade --once --venue paper` is the current bounded runtime path

## Current Observation

For the current `paper` runtime, watch these first:

- `runtime-status`
- readiness files under `data/<ASSET>/metrics/`
- `n_live_hypotheses`, `n_selected_alphas`, `n_signals_evaluated`
- `fills`, `order_failures`, and skip counts

The current bounded path is:

1. `sync-signal-cache --strict`
2. `produce-predictions --strict`
3. `trade --once --venue paper --strict`
4. `runtime-status`

## Testing

```bash
pytest tests/
ruff check src/
```

## Further Reading

- [OPERATING_BOUNDARIES.md](/home/dev/projects/alpha-os/OPERATING_BOUNDARIES.md): current runtime truth, safety boundaries, and exit criteria
- [DESIGN.md](/home/dev/projects/alpha-os/DESIGN.md): architectural rationale and long-term direction
- [ROADMAP.md](/home/dev/projects/alpha-os/docs/exploratory/ROADMAP.md): future options not yet promoted into the active plan
