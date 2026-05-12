# Fixed-State Replay Task Boundary

## Problem

`create-checkpoint-evaluation-task` is currently needed to connect a persisted
strategy checkpoint to a fixed-state replay evaluation task.

That makes the strict fixed-state OOS path depend on a manual CLI step:

- choose a source evaluation task
- choose an strategy checkpoint
- create an evaluation task
- create an evaluation job spec with `run_mode="fixed_state_replay"`

## Risk

The fixed-state replay concept is useful, but the CLI command can become the
workflow source of truth.

That is risky because the command is mostly glue:

- it does not evaluate a strategy
- it does not produce a strategy state
- it only persists the task/job-spec link needed for replay

If this stays central, alpha-os can keep accumulating CLI workflow commands
instead of making evaluation planning responsible for replay task resolution.

## Boundary

Treat `fixed_state_replay` as a current `run_mode` value, not as a target
glossary term.

Treat `create-checkpoint-evaluation-task` as a temporary adapter, not as a core
research primitive.

## Desired Direction

A fixed-state OOS run should be derivable from explicit evaluation inputs
without requiring a manual CLI command to stitch records together.

The eventual owner should be the evaluation planning path or a small domain API,
not CLI output flow.

## Close Condition

Close this when the strict fixed-state OOS golden path can create or resolve its
replay task/job spec without calling `create-checkpoint-evaluation-task`.
