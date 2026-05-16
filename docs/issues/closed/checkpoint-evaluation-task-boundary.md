# Checkpoint Evaluation Task Boundary

## Problem

`create-checkpoint-evaluation-task` was previously needed to connect a persisted
strategy checkpoint to a checkpoint-based evaluation task.

That made the strict fixed-state OOS path depend on a manual CLI step:

- choose a source evaluation task
- choose a strategy checkpoint
- create an evaluation task
- create an evaluation job spec with `strategy_checkpoint_id`

## Risk

Checkpoint-based evaluation is useful, but the CLI command could become the
workflow source of truth.

That was risky because the command was mostly glue:

- it does not evaluate a strategy
- it does not produce a strategy state
- it only persists the task/job-spec link needed for checkpoint evaluation

If this stayed central, alpha-os could keep accumulating CLI workflow commands
instead of making evaluation planning responsible for checkpoint task resolution.

## Boundary

Treat checkpoint-based evaluation as an evaluation input shape, not as a
`run_mode` value or target glossary term.

Treat `create-checkpoint-evaluation-task` as a removed temporary adapter, not as a core
research primitive.

## Desired Direction

A checkpoint-based OOS run should be derivable from explicit evaluation inputs
without requiring a manual CLI command to stitch records together.

The eventual owner should be the evaluation planning path or a small domain API,
not CLI output flow.

## Close Condition

Closed because the strict fixed-state OOS golden path now passes the checkpoint
as an explicit evaluation input without calling `create-checkpoint-evaluation-task`.
