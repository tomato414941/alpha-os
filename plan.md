# alpha-os Plan

This file is a short working plan for the current mainline.
It is not the source of truth for runtime operations. For that, prefer
`README.md`, `OPERATING_BOUNDARIES.md`, and `docs/README.md`.

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

- make the search and maintenance loop template-aware end to end

This is now more important than:

- adding a third sleeve
- tuning BTC-specific behavior
- expanding docs beyond the current map

The practical goal is:

- use `template gap` as the main missing-idea signal
- keep BTC as a bounded reference sleeve
- use ETH as the first serious non-reference validation sleeve

## Current Bottleneck

The next bottleneck is no longer basic sleeve plumbing.

The main gap is:

- search is still only partially template-aware

Right now:

- `serious` bootstrap is template-aware
- exploration is beginning to use template gaps
- but the broader search / scoring / maintenance loop is not yet fully driven
  by template coverage

## Recovery-Derived Active Work

These items came out of `OPERATING_BOUNDARIES.md` and are still active enough to deserve a
working-plan slot:

1. keep all runtime stages bounded and externally schedulable
2. do not re-enable unattended `systemd` operation yet
3. keep `trade --once --venue paper` as the canonical bounded trade entrypoint
4. continue operating from snapshots and explicit runtime status rather than
   intuition

## Highest-Value Next Steps

1. make search maintenance template-aware end to end
   - use template gaps as the main search priority
   - expose template-gap progress in sleeve snapshots over time

2. keep BTC bounded and ETH growing
   - BTC should remain a stable reference sleeve
   - ETH should continue validating non-BTC sleeve behavior

3. keep the control-plane abstraction moving
   - continue shifting from asset-specific records toward
     `template + binding + sleeve state`

## Concrete Next Tasks

### 1. Template-Gap History

- persist serious template gap snapshots in sleeve comparison reports
- make it easy to see whether a sleeve is closing or reopening template gaps

### 2. Template-Aware Search Loop

- extend guided exploration to prefer missing templates, not just missing families
- keep the priority stable across seeding and exploratory scoring

### 3. Sleeve Comparison Tightening

- compare BTC and ETH by:
  - serious template coverage
  - actionable breadth
  - backed count
  - latest serious contribution

### 4. ETH Follow-Through

- use ETH as the primary non-reference sleeve for validating:
  - template-aware search
  - serious maintenance
  - capital-path behavior

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
