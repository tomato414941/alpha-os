# Evaluation Spec Responsibility Boundary

## Problem

`EvaluationSpec` used to combine multiple concerns:

- evaluation protocol: execution range, evaluation folds, and OOS contract
- removed metric/run result configuration: `metric_windows` and
  `aggregation_kinds`

This was workable, but the name hid which part of the object owned a decision.
In ML/RL terms, evaluation protocol and metrics/logging config are related but
not the same concern.

## Risk

If more behavior keeps accumulating on `EvaluationSpec`, it may become a broad
evaluation settings bag. That would make it harder to tell whether a field is
part of the measurement protocol, run result selection, or evaluation target
selection.

## Boundary

Do not rename `EvaluationSpec` yet.

Do not split it mechanically.

## Desired Direction

Keep `EvaluationSpec` acceptable as the current persisted object, but make the
internal responsibility boundary explicit.

`EvaluationMetricConfig` was removed because it only wrapped fields already
owned by `EvaluationSpec`.

`metric_windows` and `aggregation_kinds` were removed because they were only
serialized and validated; they did not drive current evaluation behavior.

`rigor_level` was also removed because OOS behavior is now controlled directly
by `oos_contract.enforcement`.

## Non-Goals

- Do not rename `EvaluationSpec` to `EvalConfig` or `EvaluationProtocol` yet.
- Do not move universe ownership into `EvaluationSpec` as part of this issue.
- Do not change evaluation behavior before the boundary is mapped.

## Acceptance Criteria

- It is clear which fields are evaluation protocol fields.
- Metric/run result configuration is not kept on `EvaluationSpec` unless a real
  evaluation implementation uses it.
