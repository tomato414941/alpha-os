# Evaluation Spec Responsibility Boundary

## Problem

`EvaluationSpec` currently combines multiple concerns:

- evaluation protocol: execution range, evaluation folds, rigor level, and OOS
  contract
- metric/report configuration: `metric_group_names`, `metric_windows`, and
  `aggregation_kinds`
- target scoping helpers such as `target_ids`

This is still workable, but the name can hide which part of the object owns a
decision. In ML/RL terms, evaluation protocol and metrics/logging config are
related but not the same concern.

## Risk

If more behavior keeps accumulating on `EvaluationSpec`, it may become a broad
evaluation settings bag. That would make it harder to tell whether a field is
part of the measurement protocol, report selection, or evaluation target
selection.

## Boundary

Do not rename `EvaluationSpec` yet.

Do not split it mechanically. First clarify whether current fields should be
grouped as:

- evaluation protocol
- evaluation metric config
- evaluation target or scope selection

## Desired Direction

Keep `EvaluationSpec` acceptable as the current persisted object, but make the
internal responsibility boundary explicit.

The smallest likely next step is to make `EvaluationMetricConfig` a first-class
field or subdocument instead of spreading metric config fields directly across
`EvaluationSpec`.

## Non-Goals

- Do not rename `EvaluationSpec` to `EvalConfig` or `EvaluationProtocol` yet.
- Do not move universe ownership into `EvaluationSpec` as part of this issue.
- Do not change evaluation behavior before the boundary is mapped.

## Acceptance Criteria

- It is clear which fields are evaluation protocol fields.
- It is clear which fields are metric/report configuration fields.
- `target_ids` ownership is either justified on `EvaluationSpec` or moved to a
  more specific boundary issue.
