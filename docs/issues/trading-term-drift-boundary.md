# Trading Term Drift Boundary

## Problem

Some glossary entries are starting to read like implementation notes rather
than project terminology.

Examples to watch:

- `universe` can mention alpha-os usage, but should not collapse into
  "whatever the current `subject_set` table stores."
- `execution` should describe the domain concept before discussing current
  backtest mechanics.
- `benchmark` can describe how alpha-os uses comparison references, but should
  not become a dumping ground for class-name restrictions.

## Why It Matters

The glossary should remain useful as the alpha-os domain language, not a
generic trading dictionary.

But when transient schema names, migration notes, or implementation cautions
are embedded directly in definitions, readers cannot tell which parts describe
the concept and which parts describe today's implementation.

## Boundary

Glossary entries may include alpha-os-specific meaning when that meaning is
part of the domain language.

Short clarifying notes are fine. Longer naming guards, schema migration
concerns, and implementation cautions should be separated from the core
definition or moved to a boundary/design note.

## Acceptance Criteria

- Glossary entries still explain alpha-os domain usage, not just generic
  trading definitions.
- Core definitions are not primarily phrased in terms of transient table names,
  class names, migration plans, or current engine limitations.
- Project-specific cautions stay short when they appear in the glossary.
- Longer naming or implementation concerns move to a boundary/design note.

## Review Tracker

Track only terms that are likely to drift into implementation notes or overlap
with nearby alpha-os concepts. Add terms as they become relevant.

- [ ] `universe`
- [ ] `tradable universe`
- [ ] `evaluation universe`
- [ ] `subject`
- [ ] `execution`
- [ ] `execution kind`
- [ ] `strategy execution`
- [ ] `strategy execution kind`
- [ ] `run policy`
- [ ] `strategy run mode`
- [ ] `benchmark`
