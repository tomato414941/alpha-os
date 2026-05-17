# Evaluation Report Service Boundary

Status: Closed

Closed by: `evaluation_task_contract_fields.py`

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

The implementation now lives in `evaluation_task_contract_fields.py`.

- evaluation task contract extraction

## Non-Goal

Do not rename `EvaluationReport` as part of this issue. That broader persisted
artifact naming question is tracked separately.

## Close Condition

Close this when changing a strategy or portfolio field does not require editing
report code unless the stored evaluation task contract itself changes.
