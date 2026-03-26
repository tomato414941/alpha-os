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
     - greenfield architecture
     - long-horizon module boundaries
     - multi-asset / thousand-asset scaling model
   - [`portfolio-runtime-principles.md`](./portfolio-runtime-principles.md)
     - current portfolio semantics
     - lifecycle / allocation terminology
     - near-term portfolio bottlenecks and design choices
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

### "What is trusted today vs still unsafe?"

Prefer:

- [`OPERATING_BOUNDARIES.md`](../OPERATING_BOUNDARIES.md)

### "What does the current portfolio runtime mean by research/live/actionable/capital?"

Prefer:

- [`portfolio-runtime-principles.md`](./portfolio-runtime-principles.md)

### "What would the architecture look like without migration baggage?"

Prefer:

- [`DESIGN.md`](../DESIGN.md)

### "What are we doing next right now?"

Prefer:

- [`plan.md`](../plan.md)

### "How should this scale to many assets?"

Prefer:

- [`DESIGN.md`](../DESIGN.md)
  - especially the multi-asset and thousand-asset sections

### "What is an old idea, archive, or exploratory note?"

Prefer:

- [`ROADMAP.md`](./exploratory/ROADMAP.md)
- [`PREDICTION_TARGETS.md`](./exploratory/PREDICTION_TARGETS.md)
- [`TRADING_UNIVERSE_DESIGN.md`](./exploratory/TRADING_UNIVERSE_DESIGN.md)

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

### Current Design Notes

- [`portfolio-runtime-principles.md`](./portfolio-runtime-principles.md)
  - current portfolio semantics
  - sleeve lifecycle
  - allocation / actionable / capital design

### Long-Horizon Design

- [`DESIGN.md`](../DESIGN.md)
  - greenfield baseline
  - architecture without migration baggage
  - scaling model for many sleeves / assets

### Exploratory / Future-Facing

- [`ROADMAP.md`](./exploratory/ROADMAP.md)
  - forward-looking ideas
- [`PREDICTION_TARGETS.md`](./exploratory/PREDICTION_TARGETS.md)
  - target taxonomy
- [`TRADING_UNIVERSE_DESIGN.md`](./exploratory/TRADING_UNIVERSE_DESIGN.md)
  - future universe expansion ideas

### Archive / Legacy

No standalone archive design note is kept anymore.
Legacy registry-era context should be read from:

- git history
- `DESIGN.md` greenfield-vs-current sections
- `plan.md` legacy residue cleanup items

## Practical Rule

If two documents seem to disagree, prefer:

1. `README.md` for current entrypoints
2. `OPERATING_BOUNDARIES.md` for current runtime truth
3. `portfolio-runtime-principles.md` for current portfolio semantics
4. `DESIGN.md` for target architecture
5. archive / roadmap files only for context
