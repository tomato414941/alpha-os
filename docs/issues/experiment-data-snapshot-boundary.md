# Experiment Data Snapshot Boundary

## Problem

Investment hypotheses need fixed evidence data, but `data/` contains local
runtime state, caches, logs, and old outputs. It is not a reliable source of
truth for research evidence.

## Risk

If an experiment silently uses local DBs or external data, the same hypothesis
may not be reproducible later.

## Guard

Before judging a hypothesis, identify the data snapshot or data retrieval
procedure used for that hypothesis.

Do not treat `data/` as the source of truth for experiment evidence.

## Next Decision

For `crypto_regime_momentum`, decide whether the evidence data will be a
checked-in snapshot under `experiments/snapshots/` or a documented retrieval
procedure.

## Close Condition

Close this when `crypto_regime_momentum` points to exactly one evidence data
source: either a committed snapshot or a reproducible retrieval procedure.

## Later

For the first crypto hypothesis, decide whether to create a small snapshot under
`experiments/snapshots/` or keep only a retrieval procedure.
