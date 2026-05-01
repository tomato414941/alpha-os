# Experiment To Alpha-os Boundary

## Problem

alpha-os does not yet have an explicit boundary for when a promising
lightweight experiment should be moved into the main package.

## Boundary

This issue starts after a lightweight experiment has produced evidence.

It is separate from `lightweight-hypothesis-evaluation-path.md`, which is about
having a small way to test the hypothesis before using the heavier alpha-os
runtime evaluation path.

## Current Rule

Until candidate volume becomes large, use a simple rule:

- if a lightweight experiment beats its baseline, it may be moved into
  alpha-os as a candidate rule
- the moved rule must be evaluated against the same baseline through alpha-os
  artifacts
- the rule should stay as Python code with tests, not manifest DSL logic

## Risk

Without this boundary, alpha-os can drift in either direction:

- promising experiments stay outside the main evaluation path forever
- weak or unclear experiments are moved into the main package too early

## Close Condition

Close this when alpha-os has a stable, small rule for moving experiment results
into the main package without turning that rule into a large promotion system.
