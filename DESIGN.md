# Alpha-OS System Design

This file is the short entrypoint for long-horizon architecture.

It is not the source of truth for:

- current CLI entrypoints
- current scheduler policy
- current operating safety boundaries
- current short-lived execution order

For those, prefer:

- [`README.md`](./README.md)
- [`OPERATING_BOUNDARIES.md`](./OPERATING_BOUNDARIES.md)
- [`plan.md`](./plan.md)

## Design Set

Long-horizon design notes now live under [`docs/design/`](./docs/design/README.md).

Read them in this order:

1. [`docs/design/domain-model.md`](./docs/design/domain-model.md)
   - core terminology
   - hypothesis model
   - strategy hierarchy
2. [`docs/design/runtime-evaluation.md`](./docs/design/runtime-evaluation.md)
   - evaluation principles
   - targets
   - horizons
   - pipeline stages
3. [`docs/design/architecture.md`](./docs/design/architecture.md)
   - producer-consumer separation
   - prediction store contract
   - diversity and validation
4. [`docs/design/scaling-and-migration.md`](./docs/design/scaling-and-migration.md)
   - greenfield baseline
   - current repo vs target shape
   - multi-asset and large-scale scaling direction

## Design Summary

The intended architecture is:

- hypothesis-centered rather than legacy `alpha`-centered
- target-centric rather than one-horizon-by-default
- evaluation-first for predictive logic
- portfolio-level for allocation and execution outcomes
- producer-consumer separated at the prediction boundary
- scalable through template/binding/sleeve-state separation rather than
  endlessly duplicating asset-specific records

The current repository is not a pure greenfield build. It is an in-place
migration. So design work should be judged against this question:

- does the current repo move closer to the target shape while keeping legacy
  isolated from runtime truth?

## Practical Rule

If this file and another design note seem to disagree:

1. prefer the more specific file under `docs/design/`
2. prefer `README.md` for current runtime entrypoints
3. prefer `OPERATING_BOUNDARIES.md` for what is trusted today
