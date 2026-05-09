# Trading Term Drift Boundary

## Problem

Some glossary entries mix generally used trading terms with alpha-os-specific
guards or implementation vocabulary.

Examples to watch:

- `universe` should not be defined through alpha-os-only `subject` wording.
- `execution` should not be defined by contrasting it with the current
  backtest implementation.
- `benchmark` is a general comparison reference; alpha-os-specific naming
  restrictions belong outside the base definition.

## Why It Matters

The glossary should use common trading terms in their ordinary sense where the
industry meaning is stable enough.

If implementation cautions are embedded in those definitions, readers cannot
tell which parts are trading vocabulary and which parts are alpha-os policy.

## Boundary

Base term definitions should stay close to common trading usage.

Project-specific restrictions, naming guards, and schema migration concerns
should be documented separately from the base definition.

## Acceptance Criteria

- Base trading terms avoid alpha-os-only nouns unless the term is explicitly
  alpha-os-specific.
- Project-specific naming restrictions are not mixed into ordinary trading
  definitions.
- If alpha-os intentionally uses a trading term differently, the deviation is
  documented explicitly as a project-specific boundary.
