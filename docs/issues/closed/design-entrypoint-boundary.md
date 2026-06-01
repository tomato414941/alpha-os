# Design Entrypoint Boundary

Status: Closed

Closed by: root-level `DESIGN.md` has been removed as an active documentation
entrypoint. Remaining design work should be handled in focused design docs,
glossary entries, or active boundary issues.

## Problem

Root-level `DESIGN.md`, `docs/README.md`, and `docs/design/README.md`
previously overlapped as documentation entrypoints.

`DESIGN.md` mixed:

- documentation entrypoint
- long-horizon design summary
- fixed reading order
- source-of-truth rule
- stale references

## Risk

Readers cannot tell which document is current source of truth for design
questions.

Old summary text can also keep outdated framing alive after the more specific
design notes have moved on.

## Boundary

This issue is about retiring root-level `DESIGN.md` without losing potentially
useful historical design context.

It is not about rewriting all design notes, deleting historical context, or
changing runtime behavior.

New design content should go to `docs/design/`, `docs/issues/`, or
`docs/glossary.md`, depending on whether it is a durable design note, an open
boundary issue, or a term definition.

## Next Step

Root-level `DESIGN.md` has been removed as an active documentation entrypoint.
The last remaining summary is retained below for review only.

Classify each retained summary line as one of:

- delete as stale
- review before changing

## Retained Historical Summary

The removed `DESIGN.md` summary said the intended architecture was:

- research-to-evaluation lifecycle first
- signal-discovery-centered rather than legacy `alpha`-centered
- target-centric rather than one-horizon-by-default
- representation-first for large-scale predictive logic
- selection-and-compression-first for large discovery spaces
- portfolio-level for allocation and execution outcomes
- producer-consumer separated at the prediction boundary
- scalable through template/binding/sleeve-state separation rather than
  endlessly duplicating asset-specific records

It also framed the current repository as an in-place migration and asked whether
the repo moves closer to the target shape while keeping legacy isolated from
runtime truth.

## Close Condition

Close this when the retained summary has either been discarded or explicitly
accepted as still-useful historical context.
