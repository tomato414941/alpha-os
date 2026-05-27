# Evaluation Metric Group Default Boundary

## Problem

Alpha OS currently uses `metric_group_names` as the main way to describe what an
evaluation should compute.

When an evaluation spec does not provide `metric_group_names`, the system falls
back to the full `EVALUATION_METRIC_GROUP_NAMES` list. That makes an unspecified
evaluation mean "run everything".

This is too broad.

## Why This Matters

Metric groups are details of an evaluation job. They should not be the primary
way to decide what kind of evaluation is being run.

At least two different jobs are currently mixed:

- predictor evaluation: whether signal or model outputs explain a prediction
  target
- policy rollout evaluation: what happens when a trading strategy interacts
  with a market or backtest environment

Using one default metric group list makes predictor metrics such as
`prediction_diagnostics` run as part of ordinary strategy evaluation unless the
caller opts out.

## Current Finding

`EvaluationSpec.metric_group_names` defaults to `EVALUATION_METRIC_GROUP_NAMES`.

`EvaluationMetricConfig.from_document()` also defaults missing
`metric_group_names` to the full `EVALUATION_METRIC_GROUP_NAMES` list.

`EVALUATION_METRIC_GROUP_NAMES` is currently the same tuple as
`DECISION_EVALUATION_METRIC_GROUP_NAMES`, even though the list includes metrics
that are closer to predictor evaluation.

## Desired Direction

Avoid treating "no metric group specified" as "run all metric groups".

The evaluation job type should be made explicit before metric groups are chosen.
Metric groups should refine an evaluation job, not define the job by themselves.

## Close Condition

Close this when missing metric groups no longer imply a broad all-metrics
evaluation, or when the project explicitly documents and justifies that default.
