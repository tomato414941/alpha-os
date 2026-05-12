# Strategy definition source of truth

## Problem

Strategy definitions can exist in both checked-in JSON inputs and the runtime
database.

The intended flow is:

```text
JSON input
  -> apply-manifest
  -> DB runtime copy
```

But this creates a source-of-truth risk. The JSON file can look like the
canonical strategy definition, while the DB row is what the evaluation runtime
actually reads.

If both are treated as authoritative, they can drift.

## Why It Matters

For a strategy such as `crypto_regime_momentum`, the checked-in JSON contains:

```text
strategy_id
position_rule_id
portfolio
```

After applying the input, the DB stores the same strategy document in
`strategy_specs.spec_json`.

This is acceptable if the DB is clearly a runtime copy. It becomes dangerous if
the DB is edited, reused, or inspected as though it were the research source of
truth.

## Current Decision

Treat checked-in JSON inputs as the reproducible input source.

Treat DB `strategy_specs` rows as runtime artifacts derived from those inputs.

## Risk

- Updating JSON does not update an existing DB until the input is reapplied.
- A stale DB can make an evaluation look reproducible when it no longer matches
  the checked-in input.
- Lightweight hypothesis checks become heavier than necessary if every strategy
  must be written to DB before evaluation.

## Acceptance Criteria

- The project has a clear rule for when strategy definitions must be persisted
  to DB.
- Lightweight candidate checks can avoid treating DB strategy rows as canonical.
- Runtime reports can identify which checked-in input produced the DB strategy
  copy, or otherwise make the derivation explicit.
- No workflow treats both JSON and DB copies as independent sources of truth.
