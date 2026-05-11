# Run Mode Removal

## Problem

`run_mode` is a persisted implementation field with values such as
`backtest_oos`, `fixed_state_replay`, `paper`, and `live`.

The field currently behaves like an evaluation-job switch, not a stable domain
term:

- there is no first-class `RunPolicy` implementation
- `strategy run mode` duplicates the current `run_mode` field
- `backtest_oos` and `fixed_state_replay` mainly choose which inputs an
  evaluation job requires
- `paper` and `live` are reserved values, not current strategy-run domain
  concepts

## Goal

Remove `run policy` and `strategy run mode` as glossary terms, and treat
`run_mode` as a transitional implementation field to remove.

Before replacing it, decide whether a generic mode is needed at all. If the
behavior remains necessary, represent concrete evaluation job shapes and their
required inputs directly.

## Current Places To Audit

- `StrategyRunMode`
- `EvaluationJobSpec.run_mode`
- `StrategyEvaluationContext.run_mode`
- `StrategyEvaluationRequest.to_backtest_oos_run_inputs`
- `StrategyEvaluationRequest.to_fixed_state_replay_run_inputs`
- manifest and report payloads that persist `run_mode`
- validation branches keyed by `backtest_oos` or `fixed_state_replay`

## Boundary

Do not replace `run_mode` with another generic glossary term by default. The
first question is whether a mode field is needed at all.

If the behavior remains necessary, prefer explicit request shapes:

- strict OOS evaluation request inputs
- fixed-state replay evaluation request inputs
- future paper or live request inputs when those workflows actually exist

## Non-Goals

- Do not break existing manifests or stored reports without a migration path.
- Do not rename code mechanically before the request shapes are decided.
- Do not reintroduce `run policy` or `strategy run mode` as glossary terms.

## Acceptance Criteria

- New docs do not use `run policy` or `strategy run mode`.
- `run_mode` is either removed from public payloads or kept only behind a
  documented compatibility layer.
- Evaluation planning no longer depends on a single ambiguous mode if explicit
  request shapes and required inputs are available.
- Existing persisted artifacts have a migration or compatibility strategy.
