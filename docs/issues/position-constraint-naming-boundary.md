# Position constraint naming boundary

## Issue

`direction_mode` currently represents whether portfolio target weights may be
long-only, short-only, or long-short.

The name is understandable, but it is not the clearest trading term. In
portfolio and trading language, long-only / no-shorting / long-short behavior is
usually a position or portfolio constraint.

This makes `direction_mode` slightly misleading: it sounds like a generic mode,
while the field actually constrains which position directions are allowed.

## Current Decision

Do not rename it immediately.

Treat `direction_mode` as the current implementation name and
`position_constraint` as the likely target name if this field is renamed later.

`long_only` should not be a second source of truth. It should only remain, if
needed, as legacy input compatibility.

## Acceptance Criteria

- The code has one source of truth for allowed position direction.
- The persisted or public name does not make `long_only` and `direction_mode`
  look like separate strategy decisions.
- If renamed, the target name should express a constraint, not a generic mode.
