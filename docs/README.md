# Documentation Map

This file is the index for project documentation.

Use it to answer:

- which document is the source of truth for a given question
- which document is current operational truth vs design intent
- which files are archival or exploratory only

## Read Order

For most work, read in this order:

1. [`README.md`](../README.md)
   - project entry point
   - current trusted runtime path
   - quick orientation
2. [`OPERATING_BOUNDARIES.md`](../OPERATING_BOUNDARIES.md)
   - current operating posture
   - trusted / untrusted boundary
   - current safety boundaries
3. One design note, depending on the question:
   - [`DESIGN.md`](../DESIGN.md)
     - short design summary
     - entrypoint into long-horizon design notes
   - [`docs/design/README.md`](./design/README.md)
     - detailed design map
     - domain model, evaluation, architecture, scaling
4. [`plan.md`](../plan.md)
   - current working plan
   - active priority order
   - short-lived execution sequencing

## Source Of Truth By Question

### "How do I run the current system?"

Prefer:

- [`README.md`](../README.md)
- [`OPERATING_BOUNDARIES.md`](../OPERATING_BOUNDARIES.md)
- [`AGENTS.override.md`](../AGENTS.override.md) for machine-specific operations

Namespace rule:

- root CLI = current bounded runtime
- `research ...` = bounded research / offline experiments
- `legacy ...` = legacy registry-era maintenance or archive paths

### "What is trusted today vs still unsafe?"

Prefer:

- [`OPERATING_BOUNDARIES.md`](../OPERATING_BOUNDARIES.md)

### "What does the current portfolio runtime mean by research/live/actionable/capital?"

Prefer:

- [`README.md`](../README.md)
- [`OPERATING_BOUNDARIES.md`](../OPERATING_BOUNDARIES.md)
- [`runtime-evaluation.md`](./design/runtime-evaluation.md)

### "What would the architecture look like without migration baggage?"

Prefer:

- [`DESIGN.md`](../DESIGN.md)
 - [`docs/design/README.md`](./design/README.md)

### "What are we doing next right now?"

Prefer:

- [`plan.md`](../plan.md)

### "How should this scale to many assets?"

Prefer:

- [`DESIGN.md`](../DESIGN.md)
 - [`scaling-and-migration.md`](./design/scaling-and-migration.md)

### "What is an old idea, archive, or exploratory note?"

Prefer:

- [`ROADMAP.md`](./exploratory/ROADMAP.md)

## Document Roles

### Current Truth

- [`README.md`](../README.md)
  - current runtime path
  - entrypoint commands
- [`OPERATING_BOUNDARIES.md`](../OPERATING_BOUNDARIES.md)
  - current operating truth
  - safety boundaries
- [`AGENTS.override.md`](../AGENTS.override.md)
  - machine-local commands and procedures
- [`plan.md`](../plan.md)
  - current working order
  - active priority sequencing

### Long-Horizon Design

- [`DESIGN.md`](../DESIGN.md)
  - short architectural summary
- [`docs/design/README.md`](./design/README.md)
  - design note map
- [`domain-model.md`](./design/domain-model.md)
  - core terminology and hypothesis model
- [`runtime-evaluation.md`](./design/runtime-evaluation.md)
  - targets, horizons, and evaluation model
- [`architecture.md`](./design/architecture.md)
  - producer-consumer architecture
- [`scaling-and-migration.md`](./design/scaling-and-migration.md)
  - greenfield baseline and scaling direction

### Exploratory / Future-Facing

- [`ROADMAP.md`](./exploratory/ROADMAP.md)
  - future options not yet promoted into active work

### Archive / Legacy

No standalone archive design note is kept anymore.
Legacy registry-era context should be read from:

- git history
- [`scaling-and-migration.md`](./design/scaling-and-migration.md)
- `plan.md` legacy residue cleanup items

## Practical Rule

If two documents seem to disagree, prefer:

1. `README.md` for current entrypoints
2. `OPERATING_BOUNDARIES.md` for current runtime truth
3. `DESIGN.md` and `docs/design/` for target architecture
5. archive / roadmap files only for context
