# Design Notes

This directory holds the long-horizon design baseline for `alpha-os`.

Use these files when the question is architectural rather than operational:

- what the core domain model should be
- how targets, evaluations, and hypotheses relate
- how the runtime should scale beyond one bounded sleeve
- how production and consumption should be separated

Prefer these files in this order:

1. [`../../DESIGN.md`](../../DESIGN.md)
   - short design summary
   - entrypoint into the design set
2. [`domain-model.md`](./domain-model.md)
   - terminology
   - hypothesis model
   - strategy hierarchy
3. [`runtime-evaluation.md`](./runtime-evaluation.md)
   - evaluation principles
   - targets
   - horizons
   - pipeline stages
4. [`portfolio-decision.md`](./portfolio-decision.md)
   - decision layer purpose
   - portfolio inputs and outputs
   - theory-driven requirements
5. [`architecture.md`](./architecture.md)
   - producer-consumer separation
   - prediction store
   - diversity and validation
6. [`scaling-and-migration.md`](./scaling-and-migration.md)
   - greenfield vs current repo
   - multi-asset and large-scale scaling direction

This directory is not the source of truth for:

- current CLI commands
- current operating boundaries
- current near-term execution order

For those, prefer:

- [`../../README.md`](../../README.md)
- [`../../OPERATING_BOUNDARIES.md`](../../OPERATING_BOUNDARIES.md)
- [`../../plan.md`](../../plan.md)
