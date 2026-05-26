# Design Entrypoint Boundary

## Problem

`DESIGN.md`, `docs/README.md`, and `docs/design/README.md` overlap as
documentation entrypoints.

`DESIGN.md` currently mixes:

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

This issue is about the responsibility of root-level `DESIGN.md`.

It is not about rewriting all design notes, deleting historical context, or
changing runtime behavior.

Until this issue is resolved, do not expand `DESIGN.md` or add new design
content to it. New design content should go to `docs/design/`, `docs/issues/`,
or `docs/glossary.md`, depending on whether it is a durable design note, an open
boundary issue, or a term definition.

## Next Step

Classify each section of `DESIGN.md` as one of:

- keep as compatibility pointer
- move to `docs/design/README.md`
- move to `docs/README.md`
- delete as stale
- review before changing

## Close Condition

Close this when `DESIGN.md` has a clear role and no longer competes with
`docs/README.md` or `docs/design/README.md` as a source of truth.
