# Evaluation Executor Input Source Boundary

## Problem

`evaluation_execution_strategy_for_request()` used to route evaluation requests
through separate executor classes based on `execution_kind`.

That routing is suspicious because the split is not really about strategy type
or trade execution. It is mostly about where the evaluation inputs come from:

- strategy spec and subject set inputs
- signal discovery, screening, compressed belief, or strategy checkpoint
  artifacts

## Risk

Changing the routing condition from one indirect field to another, such as
`signal_discovery_id`, would remove one dependency without fixing the boundary.

The executor layer would still be deciding between input-source paths while also
constructing reports and running the common decision backtest machinery.

## Boundary

Treat executor routing as a separate concern from `execution_kind` removal.

Do not introduce a new strategy-type enum or another compatibility alias for
this split without explicit approval.

## Current Finding

The direct strategy-backtest path and the signal-discovery-artifact path differ
mainly before the common backtest machinery:

- direct path builds subject return and signal series from strategy and subject
  set inputs
- signal discovery path builds datasets from screening, compressed belief, and
  snapshot artifacts
- both paths eventually flow through range backtest variant evaluation and
  `run_decision_backtest`

`StrategyEvaluationArtifacts` currently mixes checkpoint-like references with
training or discovery provenance:

| Field | Direct path | Fixed-state path | Signal-discovery path | ML analogy |
| --- | --- | --- | --- | --- |
| `strategy_checkpoint_id` | none | present | optional | strategy checkpoint |
| `signal_discovery_run_id` | none | none | present when using a run | training run |
| `signal_discovery_id` | optional | present | present | discovery config |
| `screening_result_id` | none | present | present | selection result |
| `compressed_belief_id` | none | present | present | learned state or checkpoint component |

This makes the direct path carry a discovery-oriented artifact bundle with many
empty fields, and it makes checkpoint references hard to distinguish from
provenance references.

## Desired Direction

Clarify whether the split belongs at the dataset/input construction layer rather
than the evaluation executor layer.

## Close Condition

Close this when the evaluation executor boundary no longer hides input-source
routing behind strategy-like names, or when the current split is documented as
intentional with explicit input-source semantics.
