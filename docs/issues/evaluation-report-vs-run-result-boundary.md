# Evaluation Report vs Run Result Boundary

## Problem

`EvaluationReport` now contains mostly the persisted result of an evaluation
run, not a human-facing report.

After splitting result primitives into `evaluation_result.py`, the remaining
`EvaluationReport` wrapper mainly carries:

- `evaluation_report_id`
- `evaluation_spec_id`
- `task_results`
- `created_at`
- `evaluation_lane`
- `oos_contract_summary`
- `cross_instrument_contract`

This looks closer to an evaluation run result artifact than to a report.

## Risk

The report name can make downstream code treat persistence, display,
promotion, and audit concerns as one concept.

Current coupling includes:

- promotion decisions read candidate and baseline task results from an
  `EvaluationReport`
- baseline/adaptation provenance uses `evaluation_report_id`
- store tables and APIs are named around reports
- contract validation validates an `EvaluationReport`

These uses need an evaluation result artifact, but not necessarily a
human-facing report concept.

## Boundary

Keep `EvaluationTaskResult`, metric groups, and failure findings as evaluation
result primitives.

Treat `EvaluationReport` as a questionable wrapper name until we decide whether
the persisted artifact should become something like `EvaluationRunResult`.

Do not merge result primitives back into report code.

## Current Suspects

- `EvaluationReport`
- `evaluation_reports` store table
- `promotion_decision.py` inputs
- `source_evaluation_report_id` provenance fields

## Desired Direction

Clarify whether the persisted artifact is:

- a run result used by promotion/baseline workflows, or
- a human-facing report built from run results.

If it is a run result, prefer naming and APIs that say evaluation run/result.
Keep display/reporting as a downstream view.

## Close Condition

Close this when the persisted evaluation artifact is named and owned according
to its actual role, or when `EvaluationReport` is explicitly retained as the
project term for persisted evaluation run results.
