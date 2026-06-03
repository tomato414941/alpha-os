# Tradable Universe Code Boundary

Status: Closed. `SubjectSet` was removed from active code, so it can no longer
silently stand in for a tradable universe.

## Problem

The glossary defines `tradable universe` as the set of instruments a strategy
may hold or trade.

The code does not have a first-class `TradableUniverse` model. The previous
`SubjectSet` representation was removed from active code.

That removes the immediate risk of treating `SubjectSet` as identical to
`tradable universe`.

## Why It Matters

If a future tradable-universe object is needed, introduce it for a concrete
strategy or environment input rather than reviving `SubjectSet`.

Examples:

- selection policy may operate on tradable candidates
- feature generation may operate on observed/reference instruments
- evaluation may require a fixed evaluated set
- run result comparison may require the same tradable universe

## Boundary

Do not introduce a new `TradableUniverse` model yet.

## Non-Goals

- Do not rename existing `global_macro_tradeable_*` identifiers as part of this
  issue.

## Acceptance Criteria

- `SubjectSet` is not active code.
- Future schema or code changes should introduce tradable scope only when a
  concrete strategy or environment input needs it.
