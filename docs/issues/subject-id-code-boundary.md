# Subject ID Code Boundary

## Problem

The glossary defines `subject` as an alpha-os internal term for one thing that
can carry portfolio weight.

The code also uses `subject_id` across prediction, evaluation, screening,
belief synthesis, portfolio construction, and execution paths. This can be
valid when those layers are tracking the same portfolio-weight-bearing target,
but the boundary is implicit.

Examples of current `subject_id` usage include:

- `SubjectEvaluationInput`
- signal screening and compression records
- `SignalContribution`
- `BeliefSynthesisComponent`
- portfolio targets, construction, and execution

## Why It Matters

If `subject_id` becomes a generic prediction or evaluation target id, concepts
that cannot carry portfolio weight, such as regimes or macro states, may drift
into `subject`.

That would weaken the distinction between:

- allocation targets that can receive portfolio weight
- prediction targets that may be non-allocatable
- evaluation targets that may be subject-level, prediction-level, portfolio-level,
  or strategy-level

## Boundary

Treat `subject_id` as an allocation-target identifier unless a specific workflow
documents otherwise.

Prediction, evaluation, and belief workflows may carry `subject_id` when they
are tracking predictions or outcomes for a portfolio-weight-bearing target.

Regimes, macro states, and other non-allocatable targets should not use
`subject_id` unless they become explicit allocation targets.

## Non-Goals

- Do not rename `subject_id` immediately.
- Do not introduce a generic `target_id` replacement for all existing
  `subject_id` fields.
- Do not change persisted prediction, evaluation, or run result schemas as part of
  terminology cleanup.

## Acceptance Criteria

- Existing `subject_id` usages are classified by workflow and semantic role.
- Docs state when prediction/evaluation records are allowed to carry
  `subject_id`.
- Future non-allocatable prediction targets have a documented path that does
  not require treating them as subjects.
