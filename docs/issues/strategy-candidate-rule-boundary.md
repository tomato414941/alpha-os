# Strategy Candidate Rule Boundary

## Problem

`strategy` is too broad as a modeling term, and current code sometimes uses
`signal_kind` for rules that are not pure signals.

Examples:

- `constant_hold`
- `dual_momentum_hold`
- crypto regime momentum eligibility rules

These are closer to candidate rules than raw signals. They turn features into
eligibility, direction, or timing decisions.

## Risk

If alpha-os keeps treating candidate rules as signal kinds, the model can blur:

- signal / feature inputs
- candidate rules
- portfolio construction
- evaluation policy
- promotion decisions

That makes strategy code harder to place and encourages manifest fields to grow
into a strategy DSL.

## Desired Boundary

Use this vocabulary in new work:

- signal / feature: observable or derived input
- candidate rule: turns signals/features into eligibility, direction, or timing
- portfolio construction: sizes and combines eligible candidates
- evaluation policy: period, costs, baseline, and OOS contract
- promotion decision: promote, reject, or hold

Candidate rules should be Python code with tests, not manifest DSL logic.

## Current Follow-up

Rename the newly added crypto regime momentum function away from
`signal_series` wording before wiring it into runtime evaluation.

Do not rename the whole `signal_kind` schema until the replacement boundary is
clear from concrete candidate implementations.

## Close Condition

Close this when alpha-os has a clear naming and storage boundary for candidate
rules, without requiring strategy logic to be encoded in manifests.
