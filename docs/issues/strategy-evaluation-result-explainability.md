# Strategy Evaluation Result Explainability

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

Reports already expose strategy ids, artifact refs, strategy contract fields,
metric groups, and decision traces.

Do not add new explanatory fields until there is a concrete run result question
that cannot be answered from those facts.

## Close Condition

Close this when reports expose enough machine-readable facts to identify what
was evaluated without depending on human-facing narrative output.
