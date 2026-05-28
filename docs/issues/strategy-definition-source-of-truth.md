# Strategy definition source of truth

## Problem

Strategy definitions previously existed in checked-in JSON inputs and the
runtime database.

The old flow was:

```text
JSON input
  -> apply-manifest
  -> DB runtime copy
```

This created a source-of-truth risk. The JSON file could look like the
canonical strategy definition, while the DB row was what the evaluation runtime
actually read.

If both are treated as authoritative, they can drift.

## Why It Matters

The old checked-in JSON contained strategy-shaped records. After applying the
input, the DB stored the same strategy document in `strategy_specs.spec_json`.

Those checked-in `strategy_specs` have now been removed.

This is acceptable only if the DB is clearly a runtime copy. It becomes
dangerous if the DB is edited, reused, or inspected as though it were the
research source of truth.

## Current Decision

Treat checked-in JSON inputs as the reproducible input source.

Do not reintroduce a CLI workflow that copies strategy definitions into DB rows
before evaluation unless the DB runtime boundary is explicitly redesigned.

## Risk

- Reintroducing a DB runtime copy can make checked-in inputs and runtime rows
  drift again.
- Lightweight hypothesis checks become heavier than necessary if every strategy
  must be persisted before evaluation.

## Acceptance Criteria

- The project has a clear rule for when strategy definitions must be persisted.
- Lightweight candidate checks can avoid treating DB strategy rows as canonical.
- No workflow treats checked-in inputs and persisted copies as independent
  sources of truth.
