# Manifest naming boundary

Status: Closed

Closed by: the checked-in runtime manifest bundle directory was removed. The
remaining `examples/minimal_oos.json` is a narrow evaluation fixture, not the
primary strategy/runtime source of truth.

## Problem

`manifest` is not a very precise name for the current alpha-os JSON inputs.

The word usually suggests an inventory or package contents list. The current
files are heavier than that: they define the executable bundle used to seed a
runtime and run an evaluation.

They can include:

- subject sets
- observation bindings
- strategy specs
- evaluation specs
- evaluation target selection
- portfolio settings
- cost assumptions

This makes the name easy to underestimate. A reader may expect a lightweight
list, while the file is actually closer to an evaluation bundle.

## Current Decision

Do not rename it immediately.

`manifest` is already wired through commands, tests, examples, and checked-in
runtime configs. Renaming it without a stronger reason would create churn.

## Preferred Direction

Treat `manifest` as the current implementation name.

When the boundary becomes clearer, consider a more accurate name such as:

- `evaluation_bundle`
- `runtime_bundle`
- `evaluation_config`

`evaluation_bundle` currently describes the shape best because the file carries
multiple resources needed to reproduce an evaluation run.

## Acceptance Criteria

- The project has a clear term for the executable evaluation input bundle.
- The chosen name does not imply that the file is merely an inventory.
- The name does not make the manifest look like the full research hypothesis.
- Any rename is done only when it reduces confusion more than it adds migration
  churn.
