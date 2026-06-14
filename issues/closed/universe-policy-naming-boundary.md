# Universe Policy Naming Boundary

Status: Closed. `UniversePolicySpec` and the `SubjectSet.universe_policy`
field were removed.

## Problem

`universe_policy` used to store cross-instrument assumptions attached to a
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

The current implementation no longer carries these fields. They did not decide
which subjects belonged to the set and were not used by current behavior.

## Boundary

Do not reintroduce `universe_policy` as a generic policy/spec object.

When changing this area, distinguish at least:

- subject-set membership: which subjects are in the set
- universe construction or selection: how a set is formed
- comparison context: base currency, calendar, benchmark, and related shared
  assumptions

## Non-Goals

- Do not reintroduce compatibility fields unless a current runtime path needs
  them.

## Acceptance Criteria

- `UniversePolicySpec` is removed.
- `SubjectSet` no longer owns `universe_policy`.
- Example manifests no longer include `universe_policy`.
