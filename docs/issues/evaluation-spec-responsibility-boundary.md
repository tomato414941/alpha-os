# Evaluation Spec Responsibility Boundary

## Problem

`EvaluationSpec` currently combines multiple concerns:

- evaluation protocol: execution range, evaluation folds, and OOS contract
- metric/run result configuration: `metric_windows` and `aggregation_kinds`

This is still workable, but the name can hide which part of the object owns a
decision. In ML/RL terms, evaluation protocol and metrics/logging config are
related but not the same concern.

## Risk

If more behavior keeps accumulating on `EvaluationSpec`, it may become a broad
evaluation settings bag. That would make it harder to tell whether a field is
part of the measurement protocol, run result selection, or evaluation target
selection.

## Boundary

Do not rename `EvaluationSpec` yet.

Do not split it mechanically. First clarify whether current fields should be
grouped as:

- evaluation protocol
- evaluation metric config

## Desired Direction

Keep `EvaluationSpec` acceptable as the current persisted object, but make the
internal responsibility boundary explicit.

`EvaluationMetricConfig` was removed because it only wrapped fields already
owned by `EvaluationSpec`. The remaining question is whether `metric_windows`
and `aggregation_kinds` should stay on `EvaluationSpec` at all.

`rigor_level` was also removed because OOS behavior is now controlled directly
by `oos_contract.enforcement`.

## Non-Goals

- Do not rename `EvaluationSpec` to `EvalConfig` or `EvaluationProtocol` yet.
- Do not move universe ownership into `EvaluationSpec` as part of this issue.
- Do not change evaluation behavior before the boundary is mapped.

## Acceptance Criteria

- It is clear which fields are evaluation protocol fields.
- It is clear which fields are metric/run result configuration fields.
