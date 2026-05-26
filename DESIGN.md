# Alpha-OS System Design

Maintenance note: do not expand this file. Treat it as frozen until
`docs/issues/design-entrypoint-boundary.md` decides whether it remains only as a
compatibility pointer or is removed.

This file is the short entrypoint for long-horizon architecture.

It is not the source of truth for:

- current CLI entrypoints

For those, prefer:

- [`README.md`](./README.md)

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
