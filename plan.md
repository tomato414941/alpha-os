# alpha-os Plan

This file is a short working plan for the current mainline.
It is not the source of truth for runtime operations. For that, prefer
`README.md`, `RECOVERY.md`, and `docs/README.md`.

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
