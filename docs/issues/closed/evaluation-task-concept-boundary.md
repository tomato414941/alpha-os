# Evaluation Task Concept Boundary

Status: closed

Closed by: runtime manifests now use `evaluation_cases` /
`evaluation_case_id`, and runtime selection uses transient `EvaluationCase`
tuples instead of an `EvaluationTask` concept.

## Problem

`EvaluationTask` is not an execution task or job.

It currently stores only:

- `evaluation_case_id`
- `strategy_id`
- `evaluation_spec_id`

This makes it closer to a manifest row that says "evaluate this strategy under
this evaluation spec" than to a first-class runtime task.

The name also conflicts with ML/RL terminology. A task can mean a learning task,
an environment task, or a unit of scheduled work. The current object is none of
those.

## Risk

Keeping `EvaluationTask` as a persisted domain object encourages the evaluation
path to grow around a weak abstraction.

The current shape spreads through:

- runtime manifest field names
- store table and APIs
- CLI filters and display
- evaluation result identifiers
- documentation

Renaming it to `EvaluationCase` would improve wording, but may preserve an
unnecessary object.

## Boundary

Do not treat `EvaluationTask` as a strategy execution task.

The evaluation runtime needs:

- an `EvaluationSpec`
- one or more strategy identifiers to evaluate
- a stable result key when comparing multiple strategies

That does not necessarily require a dedicated persisted `EvaluationTask` object.

## Desired Direction

Prefer removing the standalone `EvaluationTask` concept.

Investigate moving the evaluation case list toward the evaluation spec or the
evaluation run request, while keeping strategy construction owned by strategy
definitions rather than evaluation case rows.

Avoid compatibility aliases or deprecated task interfaces unless explicitly
approved.

## Current Status

`EvaluationTask` is no longer stored in a dedicated database table, and the
input-side `EvaluationTask` class has been removed. Runtime manifests now use
`evaluation_cases` rows with optional `evaluation_case_id` fields. Those rows
are applied as strategy/evaluation cases and kept as transient
`(result_key, strategy_id)` tuples when a manifest-scoped command needs explicit
case identifiers.

## Close Condition

Closed.
