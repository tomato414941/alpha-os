# Position constraint naming boundary

Status: Closed

Closed by: current portfolio direction handling

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

`long_only` should not be a second source of truth. It may remain only as a
derived internal boolean for call paths that have not yet been renamed.

Current persisted manifests and report-facing strategy contract fields should
use `direction_mode`. Persisted/public strategy documents should not accept
`long_only` as input.

## Acceptance Criteria

- The code has one source of truth for allowed position direction.
- The persisted or public name does not make `long_only` and `direction_mode`
  look like separate strategy decisions.
- If renamed, the target name should express a constraint, not a generic mode.

## Closure Notes

`direction_mode` is the current persisted/public field for allowed position
direction. Runtime manifests use `direction_mode`.

`PortfolioConstructionSpec.from_document()` does not accept persisted
`long_only` as input. It constructs with `long_only=False` and normalizes from
`direction_mode`.

`PortfolioConstructionSpec.__post_init__()` then derives `long_only` from the
normalized `direction_mode`, so `long_only` remains an internal compatibility
boolean rather than a second source of truth.
