# Strategy Checkpoint Concept Boundary

## Problem

Alpha OS does not currently have a clean `StrategyCheckpoint` concept.

In ML/RL terms, a checkpoint is persisted fitted state that can be loaded later
without re-running training or preparation. It should represent the state needed
to run a strategy/policy after fitting.

The current `StrategyCheckpoint` implementation does not model that boundary.
It is mostly a bundle of references to signal-discovery artifacts:

- `screening_result_id`
- `compressed_belief_id`
- `snapshot_set_id`
- `survivor_signal_ids`
- `signal_discovery_id`
- fold and execution-range metadata

That makes it closer to an evaluation input bundle or provenance record than a
strategy checkpoint.

## Boundary

Do not treat the current `StrategyCheckpoint` shape as the canonical checkpoint
model.

A future checkpoint concept should answer:

- what fitted state belongs to the strategy itself
- what input data or artifact references are merely needed to reproduce it
- what metadata is provenance or diagnostics
- how a checkpoint is created independently from evaluation

## Desired Direction

Remove the current unused `StrategyCheckpoint` persistence path before it
hardens as the project checkpoint model.

Introduce a new checkpoint model later only when a fitting/preparation workflow
has a concrete persisted state to save.

## Non-Goals

- Do not remove the idea that fitted strategy state should be persisted.
- Do not define the final checkpoint schema in this issue.
- Do not make evaluation responsible for creating or discovering checkpoints.

## Acceptance Criteria

- The old signal-discovery input-bundle implementation is removed or no longer
  presented as the canonical strategy checkpoint.
- A future checkpoint design distinguishes fitted state, input references,
  provenance, and diagnostics.
