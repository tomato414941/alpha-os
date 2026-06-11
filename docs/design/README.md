# Design Notes

This directory holds the long-horizon design baseline for `alpha-os`.

Use these files when the question is architectural rather than operational:

- what the core domain model should be
- how targets, evaluations, and signals relate
- how the runtime should scale beyond one bounded sleeve
- how production and consumption should be separated

Use these files by question:

1. [`../glossary.md`](../glossary.md)
   - current terminology entrypoint
2. [`glossary.old.md`](./glossary.old.md)
   - historical terminology notes
   - signal / strategy / discovery boundary
3. [`domain-model.md`](./domain-model.md)
   - domain relationships
   - strategy hierarchy
4. [`strategy-execution-model.md`](./strategy-execution-model.md)
   - strategy vs engine boundary
5. [`runtime-evaluation.md`](./runtime-evaluation.md)
   - evaluation principles
   - targets
   - horizons
   - pipeline stages
6. [`strategy-comparison-contract.md`](./strategy-comparison-contract.md)
   - minimum facts required to compare strategy results
   - required metrics
   - optional same-subject-set check
7. [`portfolio-decision.md`](./portfolio-decision.md)
   - decision layer purpose
   - portfolio inputs and outputs
   - theory-driven requirements
8. [`portfolio-allocation-boundary.md`](./portfolio-allocation-boundary.md)
   - narrow allocator contract
   - why allocator policy should live inside implementations
   - external optimizer libraries as implementation details
9. [`architecture.md`](./architecture.md)
   - producer-consumer separation
   - prediction store
   - diversity and validation
10. [`scaling-and-migration.md`](./scaling-and-migration.md)
   - greenfield vs current repo
   - multi-asset and large-scale scaling direction

This directory is not the source of truth for:

- current CLI commands
- current operating boundaries
- current near-term execution order

For those, prefer:

- [`../../README.md`](../../README.md)
