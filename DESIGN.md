# Alpha-OS System Design

This file is the short entrypoint for long-horizon architecture.

It is not the source of truth for:

- current CLI entrypoints

For those, prefer:

- [`README.md`](./README.md)

## Design Set

Long-horizon design notes now live under [`docs/design/`](./docs/design/README.md).

Read them in this order:

1. [`docs/design/glossary.md`](./docs/design/glossary.md)
   - source of truth for terms
   - signal / strategy / discovery boundary
2. [`docs/design/constitution.md`](./docs/design/constitution.md)
   - north star
   - research-to-evaluation lifecycle
   - rigor levels and non-negotiables
3. [`docs/design/signal-discovery-system.md`](./docs/design/signal-discovery-system.md)
   - greenfield system boundary
   - signal-discovery target
   - current gap map
4. [`docs/design/domain-model.md`](./docs/design/domain-model.md)
   - domain relationships
   - strategy hierarchy
5. [`docs/design/strategy-execution-model.md`](./docs/design/strategy-execution-model.md)
   - strategy vs engine boundary
   - current mainline workflow
   - target workflow
6. [`docs/design/runtime-evaluation.md`](./docs/design/runtime-evaluation.md)
   - evaluation principles
   - targets
   - horizons
   - pipeline stages
7. [`docs/design/architecture.md`](./docs/design/architecture.md)
   - producer-consumer separation
   - prediction store contract
   - diversity and validation
8. [`docs/design/scaling-and-migration.md`](./docs/design/scaling-and-migration.md)
   - greenfield baseline
   - current repo vs target shape
   - multi-asset and large-scale scaling direction

## Design Summary

The intended architecture is:

- research-to-evaluation lifecycle first
- signal-discovery-centered rather than legacy `alpha`-centered
- target-centric rather than one-horizon-by-default
- representation-first for large-scale predictive logic
- selection-and-compression-first for large discovery spaces
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
