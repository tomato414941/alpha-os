# Evaluation Report Service Boundary

## Issue

`evaluation_report_service.py` is necessary because alpha-os needs evaluation
reports as artifacts, not just metric dictionaries.

However, the module currently mixes several responsibilities:

- report artifact construction
- strategy contract field extraction
- evaluation task contract field extraction
- subject set fact formatting
- report display ordering
- OOS contract summary formatting

This makes small strategy-schema changes, such as moving `top_k` ownership,
touch report code directly.

## Boundary

Keep the module responsible for building evaluation report artifacts.

Move or isolate responsibilities that are only input extraction or display
formatting:

- strategy contract extraction
- evaluation task contract extraction
- display-only field ordering

## Non-Goal

Do not delete `evaluation_report_service.py` just to reduce files. The report
artifact boundary is still useful.

## Close Condition

Close this when changing a strategy field does not require editing report
artifact construction unless the report schema itself changes.
