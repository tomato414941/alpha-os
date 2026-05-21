# Universe Policy Naming Boundary

## Problem

`universe_policy` currently stores cross-instrument assumptions attached to a
`SubjectSet`:

- `base_currency`
- `trading_calendar`
- `benchmark_id`

The name can imply a policy for constructing or selecting a universe, but the
current fields are closer to shared comparison/evaluation context for a
multi-subject set.

## Why It Matters

If `universe_policy` is read as "how to build the universe," future fields may
be added to the wrong object.

The current fields are used to make multi-subject validation, evaluation, and
run result comparison coherent. They do not decide which subjects belong to the set.

## Boundary

Keep the current persisted/manifest field stable until the concept is mapped.

When changing this area, distinguish at least:

- subject-set membership: which subjects are in the set
- universe construction or selection: how a set is formed
- comparison context: base currency, calendar, benchmark, and related shared
  assumptions

## Non-Goals

- Do not rename `universe_policy` immediately.
- Do not change manifest compatibility as part of terminology cleanup.
- Do not split `SubjectSet` schema until consumers and run result contracts are
  mapped.

## Acceptance Criteria

- The project has a clearer name or documented reason to keep
  `universe_policy`.
- Manifest, store, validation, run result, and CLI consumers are mapped before any
  rename.
- If renamed, old persisted/manifest documents have an explicit compatibility
  path.
