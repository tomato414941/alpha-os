# Evaluation Universe Code Boundary

## Problem

The glossary defines `evaluation universe` as the set of instruments included
in a specific evaluation run.

The code does not have a first-class `EvaluationUniverse` model or an
evaluation-universe field on `EvaluationSpec`. Current evaluation runs derive
the effective subject set from the task's strategy, strategy checkpoint, or
signal discovery run:

- trainless runs use `TradingStrategySpec.subject_set_id`
- fixed-state replay uses the strategy checkpoint's `subject_set_id`
- trained runs use the strategy checkpoint or signal discovery run
  `subject_set_id`

The resulting `subject_set_id` is then recorded in execution requests and
evaluation reports as subject-set context.

## Why It Matters

The glossary treats evaluation universe as an evaluation condition, but the
implementation currently derives it from strategy or provenance state.

That can make it hard to tell whether a subject set belongs to:

- the strategy definition
- the training/discovery provenance
- the concrete evaluation run
- the report comparison contract

If this remains implicit, the same strategy may be difficult to evaluate under a
different evaluation universe without redefining strategy or provenance inputs.

## Boundary

Do not add an `EvaluationUniverse` model yet.

Map current evaluation paths first and document where the effective evaluation
subject set comes from for each current `run_mode` value.

## Non-Goals

- Do not move universe ownership into `EvaluationSpec` immediately.
- Do not rename `subject_set_id` fields as part of terminology cleanup.
- Do not change fixed-state replay provenance behavior without a separate
  replay-specific design.

## Acceptance Criteria

- Evaluation paths document how the effective evaluation subject set is chosen.
- Reports make clear which subject set was actually evaluated.
- A future schema or API change can tell whether a subject-set field is owned by
  strategy definition, training/discovery provenance, or evaluation conditions.
