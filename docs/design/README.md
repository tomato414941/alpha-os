# Design Notes

This directory holds the long-horizon design baseline for `alpha-os`.

Use these files when the question is architectural rather than operational:

- what the core domain model should be
- how targets, evaluations, and signals relate
- how the runtime should scale beyond one bounded sleeve
- how production and consumption should be separated

Prefer these files in this order:

1. [`../../DESIGN.md`](../../DESIGN.md)
   - short design summary
   - entrypoint into the design set
2. [`glossary.md`](./glossary.md)
   - source of truth for terms
   - signal / strategy / discovery boundary
3. [`constitution.md`](./constitution.md)
   - north star
   - lifecycle stages
   - rigor levels and non-negotiables
4. [`signal-discovery-system.md`](./signal-discovery-system.md)
   - greenfield system boundary
   - signal-discovery target shape
   - current gap map
5. [`domain-model.md`](./domain-model.md)
   - domain relationships
   - strategy hierarchy
6. [`strategy-execution-model.md`](./strategy-execution-model.md)
   - strategy vs engine boundary
   - current mainline workflow
   - target execution workflow
7. [`runtime-evaluation.md`](./runtime-evaluation.md)
   - evaluation principles
   - targets
   - horizons
   - pipeline stages
8. [`portfolio-decision.md`](./portfolio-decision.md)
   - decision layer purpose
   - portfolio inputs and outputs
   - theory-driven requirements
9. [`migration-map.md`](./migration-map.md)
   - aligned vs transitional vs remove
   - module-by-module mapping
   - near-term reductions and additions
10. [`architecture.md`](./architecture.md)
   - producer-consumer separation
   - prediction store
   - diversity and validation
11. [`scaling-and-migration.md`](./scaling-and-migration.md)
   - greenfield vs current repo
   - multi-asset and large-scale scaling direction

This directory is not the source of truth for:

- current CLI commands
- current operating boundaries
- current near-term execution order

For those, prefer:

- [`../../README.md`](../../README.md)
