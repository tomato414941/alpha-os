# Signal Discovery Naming Boundary

Status: Closed

Closed by: the `SignalDiscovery` implementation was removed from the active
package. The remaining useful idea is tracked as strategy construction, not as
a `SignalDiscovery` naming problem.

## Problem

`SignalDiscovery` is easy to misread.

In the current implementation, `SignalDiscovery` is not a discovered artifact or
an executed workflow. It is closer to a search or selection specification:

- subject set
- target
- signal families
- parameter space
- screening and survivor selection policy

## Risk

The current name blurs two different concepts:

- search specification
- fitted or prepared strategy state

## Boundary

Do not treat `SignalDiscovery` as a trained strategy, model, or checkpoint.

Do not reuse the removed `StrategyCheckpoint` input-bundle shape as the
canonical checkpoint model.

## Desired Direction

Clarify the naming around signal search:

- `SignalDiscovery` should probably be renamed or documented as a search
  specification/config.

Candidate names to evaluate:

- `SignalSearchSpec`
- `SignalSelectionSpec`
- `SignalDiscoverySpec`

## Non-Goals

- Do not rename persisted fields mechanically before the new boundary is chosen.

## Acceptance Criteria

- The code and docs make it clear which object is the search specification.
- Checkpoint evaluation remains independent of workflow log records.
