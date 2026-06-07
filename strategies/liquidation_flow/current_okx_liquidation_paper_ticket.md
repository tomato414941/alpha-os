# OKX Liquidation Paper Ticket

Generated: `2026-06-07T16:04:23.344668+00:00`

This is not a trade instruction. It is a paper observation ticket.

## Candidate

- Asset: `JTO`
- Venue: `OKX USDT swap`
- Action: `long_liquidation_cascade_watch`
- Paper direction: `short`
- Paper notional: `100.00` USDT
- 15m gross continuation: `105.62` bps
- Conservative cost proxy: `12.07` bps
- Conservative net proxy: `93.55` bps
- Near-touch depth 5bps: `2207.78` USDT
- Visible depth usage: `0.0453`

## Paper Observation Shape

- Record the current mark/mid price at observation start.
- Record the simulated entry side implied by the paper direction.
- Record the 15m and 1h mark/mid price after the event timestamp.
- Subtract the same fee, spread, and depth-impact proxy used by the gate.
- Do not average into the paper position if the signal moves against the ticket.

## Falsification Checks

- Reject if the next fresh event does not reproduce the same action family.
- Reject if visible near-touch depth drops below the paper notional / 0.25 rule.
- Reject if live spread widens enough to consume the conservative net proxy.
- Reject if funding or broader perp pressure points against the paper direction.
- Reject if the 15m paper result is negative after the conservative cost proxy.

## Why This Candidate

This candidate is the strongest current liquidation paper-gate row by conservative short-window net while staying under the visible-depth usage cap. It only proves that the signal is worth paper observation; it does not prove deployable alpha.
