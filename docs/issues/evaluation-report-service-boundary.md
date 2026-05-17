# Evaluation Report Service Boundary

## Issue

`evaluation_report_service.py` is necessary because alpha-os needs evaluation
reports as artifacts, not just metric dictionaries.

The module used to mix report construction with display-time fallback logic.
After removing CLI report rendering, it now mostly builds the contract fields
stored on evaluation task results.

- evaluation task contract field extraction

This makes small strategy-schema changes, such as moving `top_k` ownership,
touch report code directly.

## Boundary

Keep the module responsible for building persisted evaluation task contract
fields.

Move or isolate responsibilities that are only strategy/portfolio extraction:

- evaluation task contract extraction

## Non-Goal

Do not delete `evaluation_report_service.py` just to reduce files. The report
artifact boundary is still useful.

## Close Condition

Close this when changing a strategy or portfolio field does not require editing
report code unless the stored evaluation task contract itself changes.
