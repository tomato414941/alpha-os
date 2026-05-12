# Run Mode Removal

## Problem

`run_mode` was a persisted implementation field with values such as
`backtest_oos`, `paper`, and `live`.

The field behaved like an evaluation-job switch, not a stable domain
term:

- there is no first-class `RunPolicy` implementation
- `strategy run mode` duplicates the current `run_mode` field
- `backtest_oos` mainly names the default evaluation job shape
- checkpoint-based evaluation is now represented by `strategy_checkpoint_id`,
  not a separate `run_mode` value
- `paper` and `live` are reserved values, not current strategy-run domain
  concepts

## Goal

Remove `run policy` and `strategy run mode` as glossary terms. Remove
`run_mode` from the active evaluation job spec.

Before replacing it, decide whether a generic mode is needed at all. If the
behavior remains necessary, represent concrete evaluation job shapes and their
required inputs directly.

## Removed Places

- `StrategyRunMode`
- `EvaluationJobSpec.run_mode`
- `StrategyEvaluationContext.run_mode` removed from the request context
- `StrategyEvaluationRequest.to_backtest_oos_run_inputs` removed
- active manifest job-spec payloads no longer persist `run_mode`
- validation branches keyed by `backtest_oos`

## Current Finding

`StrategyEvaluationContext.run_mode` was only used by request-to-run-input
helper methods and has been removed.

`EvaluationJobSpec.run_mode` has been removed. Checkpoint-based evaluation is
keyed by explicit job inputs, especially `strategy_checkpoint_id`.

## Boundary

Do not replace `run_mode` with another generic glossary term by default. The
first question is whether a mode field is needed at all.

If the behavior remains necessary, prefer explicit request shapes:

- strict OOS evaluation request inputs
- checkpoint-based evaluation request inputs
- future paper or live request inputs when those workflows actually exist

## Non-Goals

- Do not break existing manifests or stored reports without a migration path.
- Do not rename code mechanically before the request shapes are decided.
- Do not reintroduce `run policy` or `strategy run mode` as glossary terms.

## Acceptance Criteria

- New docs do not use `run policy` or `strategy run mode`.
- `run_mode` is removed from active job-spec payloads.
- Evaluation planning no longer depends on a single ambiguous mode if explicit
  request shapes and required inputs are available.
- Existing persisted artifacts are either migrated or intentionally invalidated.
