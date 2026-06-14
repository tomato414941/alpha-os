# Evaluation Universe Code Boundary

Status: Closed

## Resolution

Closed because the active package no longer has an evaluation path,
`SubjectSet`, checkpoint-owned subject-set metadata, or a first-class
evaluation-universe model.

The remaining concern is a future design topic: if evaluation universe returns,
its owner must be explicit. There is no current code boundary left to fix.

## Problem

The glossary defines `evaluation universe` as the set of instruments included
in a specific evaluation run.

The code does not have a first-class `EvaluationUniverse` model or dedicated
evaluation-universe input. `SubjectSet`, `TradingStrategySpec.subject_set_id`,
and checkpoint-owned subject-set metadata have been removed from active code.

## Why It Matters

The glossary treats evaluation universe as an evaluation condition, but active
code does not currently model it.

If evaluation universe returns, it must be clear whether the field belongs to:

- the strategy definition
- the training/preparation provenance
- the concrete evaluation run
- the run result comparison contract

If this remains implicit, the same strategy may be difficult to evaluate under a
different evaluation universe without redefining strategy or provenance inputs.

## Boundary

Do not add an `EvaluationUniverse` model yet.

Map current evaluation paths first and document where the effective evaluation
subject set comes from for each current evaluation input shape.

## Non-Goals

- Do not reintroduce a generic evaluation settings object just to hold universe
  ownership.
- Do not reintroduce `subject_set_id` fields as part of terminology cleanup.
- Do not reintroduce checkpoint-owned universe behavior without a separate
  checkpoint design.

## Acceptance Criteria

- Future evaluation paths document how the effective evaluation universe is
  chosen.
- A future schema or API change can tell whether a universe field is owned by
  strategy definition, training/preparation provenance, or evaluation
  conditions.
