# Strategy Evaluation Result Explainability

Status: Closed

## Resolution

Closed because the old evaluation report and run-result implementation has
been removed from active code.

There is currently no evaluation result object whose facts need to be repaired.
When evaluation results return, they should expose machine-readable facts about
the evaluated strategy, inputs, environment, and metrics instead of depending on
human-facing report text.

## Problem

An evaluation run result should make it possible to tell what was evaluated without
reverse-engineering execution paths or unrelated artifacts.

## Risk

If run result meaning has to be inferred from execution path names, artifact refs,
or absent fields, readers can misunderstand what a result represents.

## Boundary

This issue is about run result facts.

It is separate from:

- the archived common strategy comparison contract issue, which is about shared
  comparison conditions
- `strategy-evaluation-path-boundary.md`, which is about engine routing

## Current Finding

The old report path has been removed. Future evaluation results should expose
the facts needed to identify the evaluated strategy, inputs, environment, and
metrics without depending on human-facing report text.

Do not add new explanatory fields until there is a concrete run result question
that cannot be answered from those facts.

## Close Condition

Close this when reports expose enough machine-readable facts to identify what
was evaluated without depending on human-facing narrative output.
