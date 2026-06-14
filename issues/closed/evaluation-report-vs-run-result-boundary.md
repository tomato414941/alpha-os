# Evaluation Report vs Run Result Boundary

Status: closed

Closed by: the persisted evaluation artifact was renamed from
`EvaluationReport` to `EvaluationRunResult`.

## Resolution

The remaining wrapper now represents the stored output of one evaluation run,
not a human-facing report.

It carries:

- `evaluation_run_result_id`
- `evaluation_spec_id`
- `results`
- `created_at`
- `evaluation_lane`
- `oos_contract_summary`

The result primitive is `EvaluationResult`; `EvaluationRunResult.results` is the
map from a transient result key to the corresponding evaluation result.

## Remaining Boundary

Display/reporting should stay downstream of this object. Do not reintroduce a
core `EvaluationReport` concept unless there is a concrete reporting workflow.
