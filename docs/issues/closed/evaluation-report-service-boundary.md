# Evaluation Report Service Boundary

Status: Closed

Closed by: evaluation contract fields were removed from `EvaluationTaskResult`.

## Issue

`evaluation_report_service.py` was a misleading name for code that no longer
owned report display or report construction.

The module used to mix report construction with display-time fallback logic.
After removing CLI report rendering, it only built the contract fields stored
on evaluation task results.

- evaluation task contract field extraction

This makes small strategy-schema changes, such as moving `top_k` ownership,
touch report code directly.

## Boundary

Keep persisted evaluation task contract field extraction separate from report
display and report artifact ownership.

The intermediate `evaluation_task_contract_fields.py` helper has since been
removed. Evaluation reports no longer store a flattened strategy contract
snapshot.

- evaluation task metrics
- explicit subject set, universe policy, constraint, and sleeve metadata

## Non-Goal

Do not rename `EvaluationReport` as part of this issue. That broader persisted
artifact naming question is tracked separately.

## Close Condition

Close this when changing a strategy or portfolio field does not require editing
report code unless the stored evaluation task contract itself changes.
