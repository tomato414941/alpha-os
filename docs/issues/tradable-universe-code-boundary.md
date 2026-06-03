# Tradable Universe Code Boundary

## Problem

The glossary defines `tradable universe` as the set of instruments a strategy
may hold or trade.

The code does not have a first-class `TradableUniverse` model. Most current
flows represent the relevant set through `SubjectSet`, but `SubjectSet` also
carries subject bindings and instrument metadata.

That makes it easy to treat `SubjectSet` as identical to `tradable universe`
even when a flow only needs evaluation context or subject metadata.

## Why It Matters

If `SubjectSet` silently means tradable universe in some paths and broader
subject context in others, future strategy and evaluation code can attach fields
to the wrong object.

Examples:

- selection policy may operate on tradable candidates
- feature generation may operate on observed/reference instruments
- evaluation may require a fixed evaluated set
- run result comparison may require the same tradable universe

## Boundary

Do not introduce a new `TradableUniverse` model yet.

Map current `SubjectSet` usage first and document when it represents:

- tradable scope
- observed/reference instruments
- evaluation set
- subject metadata and bindings

## Non-Goals

- Do not rename or split `SubjectSet` immediately.
- Do not change manifest compatibility as part of terminology cleanup.
- Do not rename existing `global_macro_tradeable_*` identifiers as part of this
  issue.

## Acceptance Criteria

- Existing `SubjectSet` usages are classified by semantic role.
- Docs state when same-subject-set comparison is intended to hold tradable
  universe fixed.
- Future schema or code changes can tell whether a field belongs to tradable
  scope or broader subject-set context.
