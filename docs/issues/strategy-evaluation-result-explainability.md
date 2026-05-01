# Strategy Evaluation Result Explainability

## Problem

An evaluation report should make it possible to tell what was evaluated without
reverse-engineering execution paths or unrelated artifacts.

## Risk

If report meaning has to be inferred from execution path names, artifact refs,
or absent fields, readers can misunderstand what a result represents.

## Boundary

This issue is about report facts.

It is separate from:

- `common-strategy-evaluation-boundary.md`, which is about shared comparison
  conditions
- `strategy-evaluation-path-boundary.md`, which is about engine routing

## Current Finding

Reports already expose strategy ids, artifact refs, strategy contract fields,
metric groups, and decision traces.

Do not add new explanatory fields until there is a concrete report question
that cannot be answered from those facts.

## Close Condition

Close this when reports expose enough machine-readable facts to identify what
was evaluated without depending on human-facing narrative output.
