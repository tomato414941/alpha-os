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

## Later

For the first crypto hypothesis, decide whether to create a small snapshot under
`experiments/snapshots/` or keep only a retrieval procedure.
