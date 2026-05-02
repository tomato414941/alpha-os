# Strategy Position Rule Boundary

Resolved by making position rules explicit on `TradingStrategySpec`.

## Problem

`strategy` is too broad as a modeling term, and current code sometimes uses
`signal_kind` for rules that are not pure signals.

Examples:

- `constant_hold`
- `dual_momentum_hold`
- crypto regime momentum eligibility rules

These are closer to position rules than raw signals. They turn features into
eligibility, direction, or timing decisions.

## Risk

If alpha-os keeps treating position rules as signal kinds, the model can blur:

- signal / feature inputs
- position rules
- portfolio construction
- evaluation policy
- promotion decisions

That makes strategy code harder to place and encourages manifest fields to grow
into a strategy DSL.

## Desired Boundary

Use this vocabulary in new work:

- signal / feature: observable or derived input
- position rule: turns signals/features into eligibility, direction, or timing
- portfolio construction: sizes and combines eligible candidates
- evaluation policy: period, costs, baseline, and OOS contract
- promotion decision: promote, reject, or hold

Position rules should be Python code with tests, not manifest DSL logic.

## Resolution

`TradingStrategySpec` now stores position rule fields directly:

- `signal_discovery_id`
- `position_rule_id`
- `family_mix`
- `execution_kind`

The old `signal_policy.definition_policy` and `signal_policy.update_policy`
nesting was removed.

The duplicate sleeve-level `signal_kind` filter was removed. Sleeve signal
filtering now uses `signal_source_kind`.

Position rule implementations live in `position_rules.py`.

Position rules remain Python code with tests, not manifest DSL logic.
