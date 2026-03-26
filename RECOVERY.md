# alpha-os Recovery Plan

## Operating Posture (2026-03-21)

`alpha-os` is currently classified as `unsafe to run`.

What this means:

- do not re-enable `systemd` units yet
- do not treat manual trade runs as validation
- do not start live-trading migration work
- only run isolated tests and bounded smoke checks tied to a single recovery unit

The goal is not to "stabilize everything at once". The goal is to recover one
closed unit at a time and only then reconnect it to the runtime.

## Runtime Operating Model

The target runtime should not be a single long-lived loop that sleeps,
self-schedules, and accumulates hidden state.

The preferred operating model is:

- each runtime step is a bounded `oneshot` job
- scheduling happens outside the runtime process
- overlapping runs are rejected by lock / single-instance guarantees
- each stage records enough metadata for replay, audit, and failure analysis

### Why This Model

This is the preferred default for daily or hourly quant workflows because it:

- limits state leakage across cycles
- makes retries and rollback simpler
- localizes failures to one stage and one invocation
- avoids turning scheduler behavior into application behavior
- makes readiness measurement explicit

This does **not** rule out event-driven or always-on designs forever. It means
the first trusted runtime should be recoverable, inspectable, and restartable.

### Exception: High-Frequency / Market-Making Systems

The `oneshot` + external scheduler model is the preferred default for this
codebase, but there are legitimate exceptions.

Examples:

- high-frequency execution
- market making
- order-book reactive strategies
- event-driven systems where latency requirements are tighter than scheduler resolution

In those cases, an always-on process may be required. That exception is only
acceptable if the runtime is designed as an explicit stateful service rather
than a convenience loop.

Minimum requirements for an always-on runtime:

- explicit state machine for market / signal / execution state
- watchdogs and health probes
- kill switch / circuit breaker behavior that fails closed
- idempotent order and fill handling
- deterministic recovery after restart
- bounded memory / queue growth
- explicit handling for network partitions, stale market data, and partial execution
- independent observability for data ingest, signal generation, risk checks, and execution

In other words:

- low / medium-frequency runtime should default to bounded `oneshot` jobs
- high-frequency or MM runtime may justify always-on design
- but only with stricter operational guarantees than the default model

### Runtime Units

The runtime should be expressed as separate `oneshot` units such as:

1. `hypothesis-seeder`
2. `sync-signal-cache`
3. `produce-predictions`
4. `trade --once --venue paper`
5. readiness / status observation

Each unit should either:

- complete and emit an auditable result
- fail clearly and block downstream work
- or degrade in a pre-defined way

Do not hide all of this inside one daemon.

### Scheduling Policy

For hourly or daily operation:

- use an external scheduler (`systemd` timer, cron, or equivalent)
- call bounded `--once` entrypoints
- keep the runtime process itself stateless between invocations

In other words:

- scheduler decides **when** a cycle runs
- runtime decides **what** a cycle does

### Concurrency Policy

The same runtime profile must never run concurrently against the same asset /
venue pair.

Requirements:

- lock before trade-cycle execution
- reject overlapping invocations rather than queueing inside the process
- treat SQLite lock contention as an operational signal, not normal control flow

### Freshness And Degrade Policy

Each stage should define:

- what data freshness is acceptable
- when cached data is allowed
- when the cycle must skip rather than trade

Minimum rules:

- if required predictions are missing, do not trade
- if live hypothesis count falls below the minimum trusted set, do not trade
- if signal sync fails but cache freshness is still within policy, degrade explicitly
- if cache freshness is outside policy, fail closed

### Observability Requirements

Each cycle should record enough information to answer:

- what ran
- under which runtime profile
- with which live hypothesis set
- with which data freshness
- what was skipped
- what was traded

At minimum, persist:

- start/end timestamps
- runtime `profile_id`
- live hypothesis count
- prediction count
- selected count
- fills count
- skip / degrade reasons
- exit status

### Promotion Policy

`paper` runtime and `live` runtime should not be promoted by intuition.

Promotion path:

1. bounded local verification
2. repeated `trade --once --venue paper`
3. readiness accumulation over multiple days
4. explicit review of failures / skips / profile drift
5. only then consider live migration

For the current recovery phase, readiness matters more than uptime.

### Current Oneshot Audit (2026-03-23)

The current entrypoints are not equally ready for scheduled operation.

The draft `systemd` units accept `ALPHA_OS_CONFIG` from
`/home/dev/.secrets/alpha-os-env`. If unset, they default to the local recovery
config at `/home/dev/projects/alpha-os/config/dev.toml`.

#### `trade --once`

Current assessment: closest to the target operating model.

What it already has:

- runtime lock at the CLI entrypoint
- bounded single-cycle execution
- readiness reporting on `paper` / testnet paths
- log file output under `data/logs/`

What is still mixed in:

- scheduler mode and oneshot mode share the same command surface
- inline evolution still exists as a convenience path
- real-trading confirmation is interactive, which is correct for manual use but not for unattended scheduling

Current recommendation:

- use this as the canonical scheduled trade entrypoint
- prefer `trade --once --venue paper` for readiness accumulation

#### `paper --once`

Current assessment: bounded, but not the canonical scheduled runtime.

What it already has:

- bounded single-cycle execution
- simple output for local validation

What it is missing:

- runtime lock at the CLI boundary
- readiness integration as the primary path
- clear separation between local convenience use and scheduler-safe use

Current recommendation:

- keep this for local smoke checks and manual inspection
- do not treat it as the primary scheduled entrypoint

#### `produce-predictions`

Current assessment: structurally suitable as a stage job, but operational policy is incomplete.

What it already has:

- bounded execution
- hypothesis-native input path
- explicit count of written predictions

What it is missing:

- lock / single-writer policy
- strict failure semantics for "0 predictions written"
- explicit freshness / degrade reporting suitable for automation

Current recommendation:

- keep this as a scheduler stage
- later add stricter exit policy or a `--strict` mode

#### `sync-signal-cache`

Current assessment: bounded, but currently too lenient for unattended automation.

What it already has:

- bounded sync scope
- explicit target reporting
- cheap health check before sync

What it is missing:

- lock / single-writer policy
- explicit distinction between "degraded but acceptable" and "failed closed"
- non-zero exit behavior when required sync freshness is not met

Current recommendation:

- keep this as a scheduler stage
- later add policy-aware exit behavior for unattended runs

### Immediate Scheduling Guidance

If scheduling is introduced during recovery, the preferred chain is:

1. `sync-signal-cache`
2. `produce-predictions`
3. `trade --once --venue paper`
4. `runtime-status`

The first production-grade scheduler integration should use that chain rather
than `paper --once`.

### Recovery-Mode `systemd` Draft

The first scheduler draft should target the bounded `paper` chain only.

Suggested units under `deploy/`:

- `alpha-os-sync-signal-cache@.service`
- `alpha-os-produce-predictions@.service`
- `alpha-os-paper-trade@.service`
- `alpha-os-paper-runtime@.service`
- `alpha-os-paper-runtime@.timer`

The wrapper service should run the strict chain in order:

1. `sync-signal-cache --from-hypotheses --strict`
2. `produce-predictions --strict`
3. `trade --once --venue paper --strict`

Failure hook:

- stage units should use `OnFailure=alpha-os-runtime-status@%i.service`
- the failure hook should append `runtime-status` output into the asset log directory
- first-line triage should rely on stage exit code, stage summary line, and failure-time `runtime-status`

Important recovery rule:

- do not enable these units yet just because they exist
- use them as the canonical draft once service inventory cleanup is complete
- enable only the `paper` chain first; live trading units remain out of scope

## Code Support Matrix

The repo should not treat every historical path as equally supported.

Support classes:

- `current`: part of the hypotheses-first recovery target; keep working and test
- `research`: useful for bounded analysis, but not part of the scheduler-safe runtime
- `archive`: retained only for reference or migration context; do not extend

### CLI Classification

| Surface | Class | Rule |
|---------|-------|------|
| `hypothesis-seeder` | current | keep as a bounded runtime stage |
| `sync-signal-cache` | current | keep as a bounded runtime stage |
| `produce-predictions` | current | keep as a bounded runtime stage |
| `trade --once --venue paper` | current | keep as the canonical bounded trade entrypoint |
| `runtime-status` | current | keep as the runtime observation entrypoint |
| `testnet-readiness` | current | keep as readiness accounting for repeated bounded runs |
| `lifecycle`, `rebalance-allocation-trust`, `analyze-live-breadth` | current | keep only if they operate on `hypotheses` and remain bounded |
| `generate`, `backtest`, `evolve`, `validate`, `evaluate`, `produce-classical` | research | move under an explicit research boundary; do not treat as runtime commands |
| `paper --replay`, `replay-experiment`, `replay-matrix` | research | keep only for offline experiments until rewritten around current inputs |
| `admission-daemon`, `prune-stale-candidates`, `enqueue-discovery-pool`, `unified-generator`, `alpha-funnel` | archive | legacy registry/discovery surface; freeze and remove from the default runtime path |
| `paper --schedule`, `trade --schedule`, `paper --summary`, `trade --summary`, event-driven trade flags | archive | recovery model prefers bounded oneshot jobs; always-on convenience paths are out of scope |

### Module Classification

| Module area | Class | Rule |
|------------|-------|------|
| `hypotheses`, `predictions.store`, `data`, `execution`, `risk`, `validation.testnet`, `runtime_lock`, `runtime_profile` | current | these form the bounded recovery runtime and should be the main maintenance target |
| `paper.trader`, `paper.tracker`, `daemon.hypothesis_seeder` | current | keep only the minimum code needed for bounded paper runtime and seeding |
| `dsl`, `backtest`, `evolution`, `experiments.matrix`, `predictions.classical_producer` | research | keep available for offline work, but isolate from runtime assumptions |
| `research.replay_simulator`, `research.tactical`, `paper.event_driven` | research | keep only as lab code; they must not define scheduler or runtime truth |
| `legacy.managed_alphas`, `legacy.deployed_alphas`, `legacy.admission_replay`, `legacy.funnel`, `legacy.admission`, `legacy.lifecycle`, `legacy.alpha_generator` | archive | registry-era orchestration; no new features unless a concrete rewrite is approved |
| `research.diversity`, `research.handcrafted`, `research.replay_experiment`, `research.deployment_planner`, `research.pipeline_runner`, `research.replay_simulator`, `research.tactical` | research | bounded lab helpers and replay utilities; keep out of the runtime source of truth |
| `alpha` | compatibility shell | reserved package name only; no internal imports or wrapper modules remain |

### Alpha Package Classification

`alpha/` is not a source-of-truth package in the recovery target. It is a
retired compatibility shell kept only to reserve the package name and to
document the terminology boundary.

| File | Class | Target |
|------|-------|--------|
| `alpha/__init__.py` | compatibility shell | keep package name reserved; do not reintroduce submodules |

### Cleanup Rules

- the default `alpha-os` CLI should expose `current` commands first and treat the rest as opt-in paths
- `research` commands should move under an explicit namespace or separate entrypoint
- `archive` code must not be a dependency of the bounded runtime path
- no new source-of-truth logic should land under `alpha/`; either add it to a current package or keep it explicitly under `research` or `legacy`
- repo code, scripts, and tests should not import `alpha/`
- do not reintroduce `alpha` submodules; if a compatibility shim is truly needed again, add it only with an explicit migration rationale
- no new features should land in `archive` areas; only extraction, migration, or deletion is allowed
- if a legacy path is still needed, rewrite it around `hypotheses` inputs instead of preserving registry-era state machines

## Status Labels

- `broken`: observed runtime failure or contradiction exists
- `untrusted`: no decisive fix proof yet; runtime behavior must not be relied on
- `blocked`: depends on an earlier recovery unit
- `trusted`: explicit exit criteria met, with tests or bounded verification

## Terminology Policy

Recovery work should use `hypothesis` as the default term for a predictive unit.

- reserve `alpha` for excess return as a result
- keep legacy identifiers such as `alpha_id`, `AlphaRecord`, and
  `managed_alphas` until a dedicated rename pass
- when describing future simplified state, prefer names like
  `live hypotheses` over `deployed alphas`

## Current Architecture Snapshot

The repo is now organized around a hypotheses-first runtime with explicit
support boundaries.

### Current Runtime

- source of truth: `hypotheses.db` via `hypotheses.store.HypothesisStore`
- bounded runtime chain:
  1. `hypothesis-seeder`
  2. `sync-signal-cache`
  3. `produce-predictions`
  4. `trade --once --venue paper`
  5. `runtime-status`
- paper runtime state:
  - `paper_trading.db` stores portfolio snapshots, fills, and per-cycle hypothesis signals
  - `hypothesis_observations.db` stores realized observation history for live hypotheses
- current runtime packages:
  - `hypotheses`
  - `data`
  - `predictions`
  - `paper.trader`
  - `paper.tracker`
  - `forward.tracker`
  - `execution`
  - `risk`
  - `validation.testnet`

### Current Runtime Boundary

The current paper runtime now has a narrower internal split than earlier
recovery phases.

- `paper.trader`
  - orchestration only
  - owns the trading cycle, data sync timing, monitoring side effects, and execution
- `hypotheses.runtime_policy`
  - current selection policy
  - owns shortlist ranking, decorrelation selection, and trading signal history preparation
- `hypotheses.runtime_inputs`
  - current runtime input preparation
  - owns DSL parse validation, raw signal discovery, available-feature filtering, and prediction-history array loading

This means current runtime policy and input-prep logic should prefer
`hypotheses.*` modules over new helper methods on `paper.trader`.

### Observation Model

`hypothesis_observations.db` is part of the current runtime, but it is not a
source-of-truth registry.

- role: observation/history store for realized post-prediction outcomes
- file: `hypothesis_observations.db`
- tables:
  - `hypothesis_observations`
  - `hypothesis_observation_meta`
- typical uses:
  - live-quality estimation
  - allocation rebalance inputs
  - observation backfill
  - readiness and runtime diagnosis support

The canonical record of which hypotheses exist and what stake they carry
remains `hypotheses.db`.

### Research Boundary

Research code remains available, but it is not the runtime source of truth.

- examples:
  - `dsl`
  - `backtest`
  - `evolution`
  - `experiments`
  - `paper.simulator`
  - `paper.tactical`
  - `research.*`

Research paths may read runtime outputs, but runtime paths should not depend on
research-only orchestration.

### Legacy Boundary

Legacy code is now isolated behind the `legacy` package.

- examples:
  - `legacy.managed_alphas`
  - `legacy.registry_types`
  - `legacy.admission_queue`
  - `legacy.deployed_registry`
  - `legacy.deployed_alphas`
  - `legacy.admission_replay`
  - `legacy.admission`
  - `legacy.lifecycle`
  - `legacy.alpha_generator`
  - `legacy.funnel`

These paths may still exist for replay, migration, or archive workflows, but
they are not the hypotheses-first runtime mainline.

The remaining legacy API surface should be treated as a shrinking compatibility
layer.

- keep for now:
  - `ManagedAlphaStore`
  - `AlphaRecord`, `AlphaState`, `DeployedAlphaEntry`
  - `admission_queue`, `deployed_registry`
  - `deployed_alphas`, `admission_replay`
  - `admission`, `lifecycle`, `alpha_generator`, `funnel`
- do not add new business logic directly to `ManagedAlphaStore`
- prefer extracting pure planning or scoring logic into `research.*`
- prefer extracting table-specific I/O into focused `legacy.*` helpers
- preserve legacy entrypoints only while tests, scripts, or replay flows still depend on them

Near-term shrink candidates:

- `legacy.alpha_generator`
- `legacy.lifecycle`
- `legacy.deployed_alphas`
- `legacy.managed_alphas`
- `legacy.funnel`

Intended end state:

- `legacy.registry_types`
  - frozen compatibility data model
- `legacy.*` helpers
  - thin table-specific I/O wrappers
- no scheduler-facing or runtime-truth logic under `legacy`

### Compatibility Boundary

`alpha/` is no longer an implementation package.

- keep only `alpha/__init__.py`
- do not reintroduce `alpha` submodules
- do not add new runtime, research, or legacy logic under `alpha/`

## Target Persistent State

The target steady state is one canonical table: `hypotheses`.

Intent:

- one source of truth for all predictive units
- not DSL-specific
- no separate `deployed_alphas` table in the simplified design
- no admission queue in the simplified design
- `stake` is the runtime selection field

### Canonical Table

```sql
CREATE TABLE hypotheses (
    hypothesis_id   TEXT PRIMARY KEY,
    kind            TEXT NOT NULL,
    name            TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'active',
    stake           REAL NOT NULL DEFAULT 0.0,
    target_kind     TEXT NOT NULL DEFAULT 'forward_residual_return',
    horizon         TEXT NOT NULL DEFAULT '20D2L',
    source          TEXT NOT NULL DEFAULT '',
    scope_json      TEXT NOT NULL DEFAULT '{"universe":"core_universe_1000"}',
    definition_json TEXT NOT NULL,
    metadata_json   TEXT NOT NULL DEFAULT '{}',
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);

CREATE INDEX idx_hypotheses_status
ON hypotheses(status);

CREATE INDEX idx_hypotheses_kind
ON hypotheses(kind);

CREATE INDEX idx_hypotheses_stake
ON hypotheses(stake DESC);

CREATE TABLE hypothesis_contributions (
    hypothesis_id TEXT NOT NULL,
    date TEXT NOT NULL,
    contribution REAL NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (hypothesis_id, date)
);

CREATE INDEX idx_hypothesis_contributions_date
ON hypothesis_contributions(date);
```

### Required Columns

- `hypothesis_id`: stable identifier; should be deterministic where possible
- `kind`: how the hypothesis is represented
- `name`: short human-readable label
- `status`: operational state; start with `active` / `paused` / `archived`
- `stake`: runtime weight / keep-alive signal
- `target_kind`: canonical prediction target
- `horizon`: prediction horizon
- `source`: origin such as `random_dsl`, `manual`, `technical_library`, `ml_batch`
- `scope_json`: target asset scope; default to the shared fixed universe
- `definition_json`: executable definition payload
- `metadata_json`: non-executable annotations, audit notes, tags

The `hypotheses` and `hypothesis_contributions` tables should live in the same
SQLite database: `data/hypotheses.db`.

### Fixed Prediction Contract

For the first simplified system, all hypotheses share the same prediction
contract.

- `target_kind = forward_residual_return`
- `horizon = 20D2L`
- `scope_json = {"universe":"core_universe_1000"}`

This means:

- every hypothesis predicts the same thing
- only the *method* differs (`dsl`, `technical`, `fundamental`, `ml`, etc.)
- evaluation and trading use the same fixed universe for now
- no per-hypothesis asset-specific universe in the first pass
- newly seeded hypotheses enter as `active` with a small positive initial `stake`

Interpretation:

- not raw "up or down"
- not absolute return
- expected residual return against the common universe / benchmark surface
- Numerai-like in spirit, but simplified for this codebase

### Fixed Universe

The first simplified runtime uses one named universe:

- `core_universe_1000`

Design intent:

- fixed, shared, and versioned
- used by all hypothesis kinds
- used by both evaluation and trading in the first pass
- selected once and changed deliberately, not per run

Why this simplification:

- removes `BTC/ETH/SOL` special casing
- removes per-stage universe drift
- makes hypothesis comparison meaningful
- keeps the system closer to a Numerai-like common prediction surface

This does **not** mean the runtime must immediately trade all 1,000 assets at
full fidelity. It means the canonical hypothesis contract is defined against
that shared universe.

### Supported Kinds

- `dsl`
- `technical`
- `fundamental`
- `ml`
- `external`
- `manual`

### Definition Examples

DSL:

```json
{
  "expression": "(sub fear_greed dxy)"
}
```

Technical:

```json
{
  "indicator": "rsi_reversion",
  "params": {"window": 14, "threshold": 30},
  "inputs": ["btc_ohlcv"]
}
```

Fundamental:

```json
{
  "signal": "cpi_surprise",
  "transform": "zscore",
  "params": {"window": 20}
}
```

ML:

```json
{
  "model_type": "xgboost",
  "model_ref": "models/xgb_btc_v3.json",
  "features": ["funding_rate_btc", "oi_btc_1h", "liq_ratio_btc_1h"]
}
```

External:

```json
{
  "provider": "polymarket",
  "market": "btc-up-this-week",
  "field": "implied_probability"
}
```

Manual:

```json
{
  "rule": "long when discretionary macro regime is risk-on",
  "inputs": ["vix_close", "dxy"]
}
```

### Simplification Rules

- runtime reads from `hypotheses WHERE status = 'active' AND stake > 0`
- `stake` replaces the old deployed-set concept
- `kind` dispatches to the correct evaluator / producer
- `target_kind`, `horizon`, and `scope_json` are fixed for the first pass
- `definition_json` is the executable contract; `metadata_json` must not be required for execution
- initial inclusion is broad; later removal should happen through contribution-based stake decay

### Explicit Non-Goals

Do not add these back into the first simplified schema:

- `deployed_alphas`
- `candidates`
- state machines such as `active/dormant/rejected`
- DSL-specific columns like `expression TEXT` at the top level
- evaluation-specific fields such as `oos_sharpe`, `pbo`, `dsr_pvalue`

Those can exist in legacy tables during migration, but they are not part of the
target one-table design.

## Migration Mapping

The old tables should be treated as migration inputs, not as future design.

| Legacy source | Keep? | Mapping to `hypotheses` |
|--------------|-------|--------------------------|
| `alphas` | temporary input only | one row becomes one `hypotheses` row |
| `deployed_alphas` | remove | convert to `stake > 0` / `status='active'` semantics |
| `candidates` | remove | either discard or import as `status='paused'` only if explicitly needed |

### Legacy `alphas` -> `hypotheses`

| Legacy column | New field |
|--------------|-----------|
| `alpha_id` | `hypothesis_id` |
| `expression` | `definition_json.expression` when `kind='dsl'` |
| `state` | `status` (`active` -> `active`, `candidate/dormant/rejected` -> `paused` or dropped) |
| `stake` | `stake` |
| `created_at` | `created_at` |
| `updated_at` | `updated_at` |
| `metadata` | `metadata_json` |

Fields such as `fitness`, `oos_sharpe`, `pbo`, and `dsr_pvalue` are legacy
evaluation baggage. Do not carry them into the first simplified schema.

### Legacy `deployed_alphas` -> `hypotheses`

`deployed_alphas` should not survive as a separate table.

Migration rule:

- if an item is deployed, the hypothesis remains `active`
- deployed membership should influence initial `stake` during migration if needed
- after migration, runtime selection is entirely `stake`-based

### Legacy `candidates` -> `hypotheses`

Default rule:

- do not migrate `candidates`

Reason:

- the simplified design removes admission queues
- pending rows without runtime trust add complexity without value

If a recovery migration needs to preserve manual ideas, import them as:

- `status = 'paused'`
- `stake = 0`
- `kind = 'dsl'` or `kind = 'manual'`
- `source = 'legacy_candidate_import'`

### Initial Runtime Rule After Migration

Immediately after the first migration, runtime should read only:

```sql
SELECT *
FROM hypotheses
WHERE status = 'active' AND stake > 0
ORDER BY stake DESC;
```

## Recovery Units

| Unit | Status | Why it is in this state | Exit criteria |
|------|--------|--------------------------|---------------|
| order-constraint-boundary | broken | Observed runtime failure in venue precision path (`SOL/USDT` precision error in `constrain_intent()` flow) | Invalid venue quantities are rejected before submit; no exception escapes from intent constrainting; regression tests cover the observed failure path |
| managed-state-accounting | untrusted | Internal cash/position tracking may drift after fills; later checks depend on this state being correct | Fill application rules are deterministic and covered by tests for buy/sell/partial fill cases |
| reconciliation | broken | Observed BTC/ETH/SOL mismatch between internal state and exchange state | Baseline semantics are documented; isolated tests cover clean-account and shared-account cases; latest bounded run reports `match=True` |
| runtime-reporting-readiness | untrusted | `runtime-status` reports profile/report drift and stale readiness context | Report generation uses one clear source of truth; profile/deployed-set identifiers align with the current run; stale-report warnings disappear in bounded verification |
| scheduler-service-lifecycle | broken | Services are disabled, naming drift existed, and manual single-run traces were mixed into runtime observation | Unit inventory is cleaned; one canonical run path exists; restart/stop/status commands match docs; bounded service smoke check passes |
| generator-discovery-pipeline | untrusted | README already marks generator side as needing repair; current trust is insufficient for continuous runtime | Generator health metrics are defined; archive/admission flow passes bounded smoke checks; failures are observable from one command |
| ops-docs-boundary | untrusted | Current state was spread across README, override notes, logs, and ad hoc observations | Runtime posture, repair order, and restart conditions are documented in one place and referenced from entry docs |

## Recovery Order

Work in this order. Do not skip ahead unless the current unit is explicitly
blocked by external dependency work.

1. `order-constraint-boundary`
2. `managed-state-accounting`
3. `reconciliation`
4. `runtime-reporting-readiness`
5. `scheduler-service-lifecycle`
6. `generator-discovery-pipeline`
7. `ops-docs-boundary`

## First Unit: order-constraint-boundary

This is the first focus because it is the smallest closed unit with direct
runtime impact.

Scope:

- `src/alpha_os/execution/binance.py`
- `src/alpha_os/execution/constraints.py`
- `tests/test_binance_executor.py`
- `tests/test_execution.py`

Non-goals:

- reconciliation redesign
- `systemd` recovery
- generator repairs
- readiness/reporting cleanup beyond what this unit strictly needs

Definition of done:

- `ExecutionIntent -> constrain_intent()` never crashes on venue precision errors
- venue-invalid quantities return `order=None` with an explicit rejection reason
- min-notional and precision application order is consistent
- a regression test reproduces the observed `SOL` failure mode
- targeted tests pass

Suggested verification:

```bash
pytest tests/test_binance_executor.py tests/test_execution.py -q
```

## Restart Gate

`alpha-os` remains `unsafe to run` until at least these units are `trusted`:

- `order-constraint-boundary`
- `managed-state-accounting`
- `reconciliation`
- `runtime-reporting-readiness`
- `scheduler-service-lifecycle`

Before that point, any runtime execution should be treated as debugging only,
not as evidence of recovery.
