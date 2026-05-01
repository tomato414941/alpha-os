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
2. One design note, depending on the question:
   - [`DESIGN.md`](../DESIGN.md)
     - short design summary
     - entrypoint into long-horizon design notes
   - [`docs/design/README.md`](./design/README.md)
     - detailed design map
     - domain model, evaluation, architecture, scaling
## Source Of Truth By Question

### "How do I run the current system?"

Prefer:

- [`README.md`](../README.md)
- `AGENTS.override.md` for machine-specific operations when present locally

Namespace rule:

- root CLI = current bounded runtime

### "What does the current portfolio runtime mean by decision state?"

Prefer:

- [`README.md`](../README.md)
- [`runtime-evaluation.md`](./design/runtime-evaluation.md)

### "What would the architecture look like without migration baggage?"

Prefer:

- [`DESIGN.md`](../DESIGN.md)
- [`docs/design/constitution.md`](./design/constitution.md)
- [`docs/design/README.md`](./design/README.md)

### "How should this scale to many assets?"

Prefer:

- [`DESIGN.md`](../DESIGN.md)
- [`scaling-and-migration.md`](./design/scaling-and-migration.md)

### "When are two strategy results comparable?"

Prefer:

- [`strategy-comparison-contract.md`](./design/strategy-comparison-contract.md)

## Document Roles

### Current Truth

- [`README.md`](../README.md)
  - current runtime path
  - entrypoint commands
- `AGENTS.override.md`
  - machine-local commands and procedures

### Long-Horizon Design

- [`DESIGN.md`](../DESIGN.md)
  - short architectural summary
- [`docs/design/README.md`](./design/README.md)
  - design note map
- [`constitution.md`](./design/constitution.md)
  - north star
  - lifecycle stages
  - rigor levels and non-negotiables
- [`glossary.md`](./design/glossary.md)
  - source of truth for terminology
- [`domain-model.md`](./design/domain-model.md)
  - domain relationships
- [`strategy-execution-model.md`](./design/strategy-execution-model.md)
  - strategy vs engine boundary
  - current and target workflows
- [`runtime-evaluation.md`](./design/runtime-evaluation.md)
  - targets, horizons, and evaluation model
- [`strategy-comparison-contract.md`](./design/strategy-comparison-contract.md)
  - minimum comparable-result contract
- [`architecture.md`](./design/architecture.md)
  - producer-consumer architecture
- [`scaling-and-migration.md`](./design/scaling-and-migration.md)
  - greenfield baseline and scaling direction

### Archive / Legacy

No standalone archive design note is kept anymore.
Legacy registry-era context should be read from:

- git history
- [`scaling-and-migration.md`](./design/scaling-and-migration.md)

## Practical Rule

If two documents seem to disagree, prefer:

1. `README.md` for current entrypoints
2. `DESIGN.md` and `docs/design/` for target architecture
