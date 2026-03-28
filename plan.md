# alpha-os Plan

This file is a short working plan for the current mainline.
It is not the source of truth for runtime operations. For that, prefer
`README.md`, `OPERATING_BOUNDARIES.md`, and `docs/README.md`.

## V1 Completion Contract

This section is the source of truth for the next greenfield-style build.

- Scope: `1 asset`, `paper-only`, `1 target`, `1 end-to-end cycle`
- Required path: `prediction -> observation -> metrics update -> snapshot`
- Execution can stop at virtual portfolio weights; real order routing is not required for v1
- Runtime state must have a single source of truth; do not split the canonical cycle state
- Every artifact in one cycle must share the same `evaluation_id`
- Multi-asset, global allocator, live trading, and legacy compatibility are out of scope for v1
- New tables or runtime states require proof that the current model cannot express them
- New improvement ideas go to a `post-v1 backlog`; do not merge them into the v1 path mid-build
- Weekly review question: "did v1 get closer to end-to-end completion?"
- Until end-to-end completion exists, prefer duplication over premature abstraction

### V1 Done When

V1 is complete when all of the following are true:

- `generate-evaluation-inputs -> apply-backfill -> status -> show-evaluations` works as one bounded end-to-end path
- the v1 smoke test fixes that path as the completion gate
- `evaluation_snapshots` preserve enough provenance to audit input source and date range
- v1 remains within the contract scope: `1 asset`, `paper-only`, `1 target`, `1 cycle model`
- new ideas that do not strengthen that bounded path are kept out of the v1 runtime

### Post-V1 Backlog Boundary

The following belong to `post-v1 backlog`, not the v1 runtime:

- multi-asset support
- global or cross-sleeve allocation
- live trading
- legacy compatibility layers
- broader research abstractions that do not improve the bounded v1 cycle

## V2 Rules

This section defines the next step after v1 freeze.

- Scope stays small: `BTC-only`, `paper-only`, `1 target`
- V2 focus: make the runtime explicitly `hypothesis-first`
- Add explicit stages for:
  - `register-hypothesis`
  - `record-prediction`
  - `finalize-observation`
  - `update-state`
- Keep the v1 bounded cycle intact; v2 may refactor the path, but must not remove auditability
- `evaluation_id` remains mandatory across all cycle artifacts
- Do not add multi-asset, global allocation, or live routing in v2
- Do not redesign the scoring model and lifecycle model in the same step
- V2 is done when hypothesis registration is explicit and the same end-to-end audit path still works

### V2 Done When

V2 is complete when all of the following are true:

- `register-hypothesis -> record-prediction -> finalize-observation -> update-state` works as the primary bounded path
- the v2 smoke test fixes that primary path as the completion gate
- `apply-evaluation` and `apply-backfill` remain available as convenience wrappers over that same path
- `evaluation_id` and per-evaluation provenance remain auditable through `show-evaluations`
- missing or conflicting prediction / observation records fail explicitly
- v2 stays within scope: `BTC-only`, `paper-only`, `1 target`

### Still Not V2

The following are still out of scope after v2:

- multi-asset runtime
- global allocation
- live execution
- richer hypothesis lifecycle states beyond `registered` and `live`
- scoring-model redesign beyond the current bounded update rules

## V3 Rules

This section defines the next step after v2 completion.

- Scope still stays small: `BTC-only`, `paper-only`, `1 target`
- V3 focus: make the hypothesis lifecycle an explicit state machine
- Keep the primary bounded path:
  - `register-hypothesis`
  - `record-prediction`
  - `finalize-observation`
  - `update-state`
- Add explicit hypothesis states:
  - `registered`
  - `active`
  - `live`
  - `paused`
  - `retired`
- Distinguish `live hypothesis` from `live trading`
  - `live hypothesis` means allocation-eligible inside the runtime
  - `live trading` remains out of scope
- Reject invalid transitions explicitly rather than silently repairing them
- Keep `apply-evaluation` and `apply-backfill` as convenience wrappers over the same bounded path
- Do not add multi-asset, global allocation, or live execution in v3
- Do not redesign the scoring model and the lifecycle state machine in the same step

### V3 State Machine

The intended state flow is:

- `register-hypothesis` creates `registered`
- `update-state` may move `registered -> active`
- allocation eligibility may move `active -> live`
- operator action may move `live -> paused`
- operator action may move `paused -> active`
- operator action may move `active -> retired`
- operator action may move `paused -> retired`

### V3 Done When

V3 is complete when all of the following are true:

- lifecycle states are explicit in the runtime model
- valid transitions are enforced and invalid transitions fail explicitly
- the primary bounded path still works and remains auditable
- `live hypothesis` semantics are explicit without introducing live trading
- v3 stays within scope: `BTC-only`, `paper-only`, `1 target`

## V4 Rules

This section defines the next step after v3 completion.

- Scope still stays small: `BTC-only`, `paper-only`, `1 target`
- V4 focus: separate transition policy from scoring and state storage
- Keep the existing bounded path and lifecycle state machine intact
- Make transition decisions explicit policy outputs rather than implicit side effects
- The policy layer may decide:
  - when `active -> live`
  - when `live -> paused`
  - when `active -> retired`
  - when `paused -> active`
- Runtime storage and CLI should enforce transitions, not invent them ad hoc
- Do not add multi-asset, global allocation, live execution, or new read-only surfaces in v4
- Do not redesign the scoring model and transition policy in the same step

### V4 Done When

V4 is complete when all of the following are true:

- transition policy is explicit and separate from persistence
- state transitions are driven by named policy decisions rather than embedded conditions
- the bounded path remains auditable and deterministic
- v4 stays within scope: `BTC-only`, `paper-only`, `1 target`

## V5 Rules

V5 focuses on state minimization rather than new state introduction.

- Keep scope fixed: `BTC-only`, `paper-only`, `1 target`
- Do not add new persisted state just to make the runtime easier to inspect
- Separate `persisted state` from `derived label`
- Re-evaluate hypothesis states one by one:
  - `registered`
  - `active`
  - `live`
  - `paused`
  - `retired`
- Keep a state only when one of the following is true:
  - it captures operator intent that cannot be derived later
  - it enforces a transition constraint that cannot be expressed safely from existing records
  - it marks an irreversible boundary that should survive recomputation
- Prefer derived policy labels when a value can be recomputed safely from:
  - recorded predictions
  - finalized observations
  - allocation trust
  - cycle snapshots
- Treat `active` and `live` as removal candidates first
- Do not add cycle lifecycle states in v5 unless existing records prove insufficient
- Do not add multi-asset, global allocation, live execution, or new read-only surfaces in v5

V5 is complete when all of the following are true:

- each persisted hypothesis state has an explicit justification
- removable states have either been deleted or defended with a concrete constraint
- derived labels are computed by policy rather than stored by default
- the bounded path remains auditable and deterministic after any simplification

## Current Position

- hypotheses-first runtime is established
- BTC is the reference sleeve
- ETH is the first non-BTC sleeve
- `serious` seeds are template-driven
- `run-sleeves-once` supports independent serious maintenance
- reference sleeve growth is capped
- serious template coverage is visible per sleeve

## What Has Been Proven

- hypothesis registration is sleeve-aware
- serious maintenance is an independent stage, not a seeding side effect
- BTC and ETH can run in the same bounded sleeve loop
- serious seeds can pass:
  - research
  - live
  - actionable
  - capital
- the reference sleeve can be kept bounded while non-reference sleeves keep growing

## Immediate Project-Wide Priority

The current project-wide priority is:

- define sleeve-level control metrics and drive search / maintenance / capital
  attention from them

This is now more important than:

- adding a third sleeve
- tuning BTC- or ETH-specific behavior
- expanding docs beyond the current map

The practical goal is:

- use a small set of sleeve-general control metrics rather than one local gap
  signal
- keep BTC as a bounded reference sleeve
- use ETH as the first serious non-reference validation sleeve

## Current Bottleneck

The next bottleneck is no longer basic sleeve plumbing.

The main gap is:

- control policy is still too local and too template-gap-centric

Right now:

- `serious` bootstrap is template-aware
- exploration and reporting can see template gaps
- but the broader search / scoring / maintenance loop is not yet driven by a
  stable sleeve-general control metric

## Recovery-Derived Active Work

These items came out of `OPERATING_BOUNDARIES.md` and are still active enough to deserve a
working-plan slot:

1. keep all runtime stages bounded and externally schedulable
2. do not re-enable unattended `systemd` operation yet
3. keep `trade --once --venue paper` as the canonical bounded trade entrypoint
4. continue operating from snapshots and explicit runtime status rather than
   intuition

## Highest-Value Next Steps

1. define sleeve control metrics
   - formalize a small set of sleeve-general metrics such as:
     - coverage retention
     - capital conversion
     - breadth trend
   - make these the primary control-plane signals rather than raw
     asset-specific symptoms

2. drive budgets from those metrics
   - use sleeve control metrics for:
     - search budget
     - serious maintenance intensity
     - rebalance attention
   - keep `template gap` as one input, not the whole policy

3. keep BTC bounded and ETH growing
   - BTC should remain a stable reference sleeve
   - ETH should continue validating non-BTC sleeve behavior

4. keep the control-plane abstraction moving
   - continue shifting from asset-specific records toward
     `template + binding + sleeve state`

## Concrete Next Tasks

### 1. Sleeve Control Metrics

- define sleeve-level control metrics that represent:
  - whether ideas are being retained
  - whether retained ideas convert into backed sleeves
  - whether actionable breadth is improving or degrading
- make those metrics the canonical control inputs

### 2. Budget Policy Rework

- update budget policy so it is not driven by `template gap` alone
- use `template gap` as one component of a broader sleeve score
- avoid asset-specific rescue logic

### 3. Sleeve Comparison Tightening

- compare BTC and ETH by:
  - control metrics
  - search budget
  - serious template coverage
  - backed count and breadth deltas

### 4. ETH Follow-Through

- use ETH as the primary non-reference sleeve for validating:
  - sleeve-general control policy
  - serious maintenance under shared rules
  - capital-path behavior without manual rescue

### 5. Legacy Residue Removal

The old registry-era design note is no longer the right place to track these.
The remaining residue should be handled as active cleanup work here:

- remove remaining `legacy/` runtime dependencies from the current mainline
- keep `legacy admission-daemon`, `managed_alphas`, and `deployed_alphas` out of the
  trusted sleeve runtime
- continue replacing `alpha_id` / registry-era naming in non-legacy paths
- isolate or retire research helpers that still depend on `alpha_registry.db`
- keep legacy replay and archive workflows readable, but not on the current
  runtime path

Recent progress:

- moved `pipeline_runner`, `replay_experiment`, and `replay_simulator` under
  `legacy/`
- moved `deployment_planner` and `registry_signal_map` under `legacy/`
- moved legacy replay / admission command handlers into `legacy.cli_commands`
- `cli.py` now acts as a thinner dispatcher for those legacy surfaces

## Notes Moved From Design Docs

These items live here on purpose because they are expected to change often:

- current sleeve transition status
- immediate project-wide priority
- current "what to do next" ordering

They do not belong in:

- `DESIGN.md`
  - which should stay greenfield and long-horizon
- `docs/portfolio-runtime-principles.md`
  - which should stay focused on portfolio semantics and near-term design rules

## Explicit Non-Priorities

Do not prioritize these right now:

- adding a third sleeve
- long-short expansion
- large refactors for naming alone
- broad documentation expansion beyond the current map
- asset-specific rescue logic
- adding more local snapshot / comparison helpers without improving the control
  policy
- treating `template gap` as the sole objective

## Remove / Simplify

The following should be reduced or treated as secondary rather than primary:

- `template gap` as a standalone control objective
  - keep it as one useful signal, but not the main policy target
- asset-by-asset tactical tuning
  - use BTC and ETH as validation sleeves, not as special-case policy inputs
- control logic spread across CLI output helpers
  - continue moving intermediate policy and delta logic into services
- observation-only feature additions that do not improve control decisions
  - prefer changes that improve allocation, search, or maintenance decisions

## Scaling Direction

The long-term scaling direction remains:

- `template`
- `binding`
- `sleeve state`

The goal is not to stretch the current per-asset loop to 1000 assets.
The goal is to keep introducing the right abstractions so a later batched
control plane becomes possible.

## Greenfield Target-Centric Build

If the project were rebuilt from scratch to support a broader predictive OS,
the implementation plan should be target-centric rather than
hypothesis-centric.

The intended target classes are not only market-facing targets such as:

- directional return
- volatility / regime
- cross-sectional relative return
- correlation / structure
- liquidity / microstructure
- sentiment / macro
- on-chain

They also include operational targets such as:

- hypothesis / model health
- allocation / shortlist inclusion
- execution / signal persistence
- runtime data trust

### Phase 1. Fix The Domain Model

Define these first-class objects:

- `target_template`
- `target_binding`
- `predictor_template`
- `predictor_binding`
- `prediction_record`
- `observation_record`
- `decision_record`

Do not collapse predictor and target into the same source-of-truth record.

### Phase 2. Define Runtime Contracts Per Target

For each target class, define:

- prediction schema
- observation schema
- scoring metric
- actionable semantics
- capital semantics

Examples:

- `market.directional`
- `market.volatility_regime`
- `market.cross_sectional`
- `hypothesis.decay_risk`
- `execution.signal_persistence`

### Phase 3. Build Target-Centric Storage

Use storage that centers:

- templates
- bindings
- predictions
- observations
- scores
- allocation / execution decisions

Avoid an asset-specific hypothesis record as the primary abstraction.

### Phase 4. Build A Batched Data Plane

Evaluation should run by:

- target template
- predictor template
- asset batch

not by one sleeve loop at a time.

### Phase 5. Split The Control Plane

Keep these responsibilities separate:

- target registry
- predictor registry
- evaluation service
- decision service
- portfolio allocator

CLI should orchestrate these services, not contain their policy.

### Phase 6. Start With A Small Target Set

The first greenfield targets should be:

1. `market.directional`
2. `market.volatility_regime`
3. `hypothesis.actionable_probability`
4. `execution.signal_persistence`

Do not start with every target class at once.

### Phase 7. Make Scoring Target-Specific

Examples:

- directional: Sharpe, calibration, hit-rate
- cross-sectional: IC, rank IC
- decay: Brier, AUC
- execution persistence: calibration, realized retention

Then derive a common meta-score from:

- predictive quality
- actionability
- capital value

### Phase 8. Make Allocation Three-Level

Allocation should happen at:

1. predictor within target
2. target within sleeve
3. sleeve within global portfolio

This is the level required for serious multi-target and thousand-asset scale.

### Phase 9. Keep Explanations As First-Class State

Every retained / rejected / traded decision should write reasons such as:

- why retained
- why actionable
- why not capital-backed
- why not traded

This should be decision log state, not only diagnostics.

### Phase 10. Rollout Order

Implement in this order:

1. domain model and schema
2. `market.directional`
3. `market.volatility_regime`
4. `hypothesis.actionable_probability`
5. `execution.signal_persistence`
6. sleeve allocation
7. `market.cross_sectional`
8. global portfolio allocation

### Greenfield Non-Goals

Do not repeat these mistakes in a greenfield build:

- overloading `stake` with multiple meanings
- treating predictor and target as the same record
- growing from single-asset assumptions and retrofitting multi-asset later
- using one scoring philosophy for every target class
- placing policy in CLI entrypoints
