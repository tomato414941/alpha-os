# Alpha-OS System Design

Maintenance note: do not expand this file. Treat it as frozen until
`docs/issues/design-entrypoint-boundary.md` decides whether it remains only as a
compatibility pointer or is removed.

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
