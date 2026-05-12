# Signal Discovery Naming Boundary

## Problem

`SignalDiscovery` and `SignalDiscoveryRun` are easy to misread.

In the current implementation, `SignalDiscovery` is not a discovered artifact or
an executed workflow. It is closer to a search or selection specification:

- subject set
- target
- signal families
- parameter space
- screening and survivor selection policy

`SignalDiscoveryRun` is also not an evaluation input source. It is closer to a
log for one execution of that search workflow, including runtime and selection
counts.

## Risk

The current names blur three different concepts:

- search specification
- workflow execution log
- prepared checkpoint used for evaluation

That confusion already leaked into evaluation planning: evaluation used to read
`SignalDiscoveryRun` records as a fallback source for prepared inputs, and
`StrategyCheckpoint` used to carry `signal_discovery_run_id` as required
provenance.

## Boundary

Do not treat `SignalDiscovery` as a trained strategy, model, or checkpoint.

Do not make `SignalDiscoveryRun` a source of truth for evaluation inputs.

Keep `StrategyCheckpoint` as the fixed state used for evaluation.

## Desired Direction

Clarify the naming around signal search:

- `SignalDiscovery` should probably be renamed or documented as a search
  specification/config.
- `SignalDiscoveryRun` should probably be renamed or documented as a workflow
  execution log or diagnostics record.

Candidate names to evaluate:

- `SignalSearchSpec`
- `SignalSelectionSpec`
- `SignalDiscoverySpec`
- `SignalSearchRunLog`
- `SignalDiscoveryRunLog`

## Non-Goals

- Do not rename persisted fields mechanically before the new boundary is chosen.
- Do not reintroduce discovery-run provenance into `StrategyCheckpoint`.
- Do not make training/run diagnostics part of checkpoint evaluation metrics by
  default.

## Acceptance Criteria

- The code and docs make it clear which object is the search specification.
- The code and docs make it clear which object is the execution log.
- Evaluation planning does not use discovery-run records as prepared input
  sources.
- Checkpoint evaluation remains independent of discovery-run records.
