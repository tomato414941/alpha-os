# Live Execution And Data Collection Boundary

Status: Closed

Closed by: current README and signal-noise design boundary

## Problem

alpha-os has evaluation-time execution cost models, signal-noise adapters, and
some backfill helpers, but it should not become the live trading system or the
data collection system.

The current project scope is hypothesis research, strategy candidate
backtesting and OOS evaluation, signal screening, and decision records.

## Risk

If alpha-os owns live execution, exchange connectors, and data collectors, the
project expands from research evaluation into operations infrastructure.

That would pull work toward:

- order submission
- fills and account reconciliation
- exchange-specific connector maintenance
- vendor-specific data collection
- runtime monitoring and incident handling

Those responsibilities can block the smaller goal of judging strategy
candidates.

## Boundary

alpha-os may own:

- execution cost models
- turnover and transition models
- evaluation-time data adapters
- signal-noise consumption for evaluation inputs

alpha-os should not own:

- live order execution
- broker or exchange account sync
- exchange connector implementations
- data vendor or exchange collector implementations

## Close Condition

Close this when live execution, exchange connector, and data collection
responsibilities are either explicitly out of scope or owned by separate
projects, while alpha-os keeps only the evaluation-facing interfaces it needs.

## Closure Notes

The root `README.md` scopes the current mainline to signal discovery research,
strategy definition, OOS evaluation, and portfolio decision/evaluation flows.
It does not include live order execution, exchange account sync, or exchange
connector ownership.

`docs/design/constitution.md` states that `signal-noise` is a separate data
service. alpha-os owns adapters, contracts, and validation for the data it needs
from that service, while signal-noise owns its own data product.

This leaves alpha-os responsible for evaluation-facing interfaces and keeps live
execution, exchange connectors, and data collection outside the current project
scope.
