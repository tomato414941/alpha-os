# OKX-Hyperliquid Paper Ticket

Generated: `2026-06-07T11:56:42.308004+00:00`

This is not a trade instruction. It is a paper feasibility ticket.

## Candidate

- Asset: `BTC`
- Long venue: `OkxSwap`
- Short venue: `HlPerp`
- Persistence observations: `3`
- Positive 8h net rate: `1.0000`
- Mean annualized spread: `0.13314743`
- Mean 8h net proxy: `0.00009284`
- Min 8h net proxy: `0.00007061`
- Max 8h net proxy: `0.00010416`
- Mean 24h net proxy: `0.00033603`
- Mean breakeven holding time: `1.8913` hours
- Mean capacity proxy notional: `764336.86`
- Paper notional cap: `1000.00` USDT

## Paper Order Shape

- Leg 1: open a long `BTC` perp exposure on `OkxSwap`.
- Leg 2: open a short `BTC` perp exposure on `HlPerp`.
- Use equal notional on both legs.
- Use paper/notional-only tracking until venue order constraints are verified.
- Target notional is capped by the smaller of 1,000 USDT and 1% of capacity proxy.

## Falsification Checks

- Confirm OKX and Hyperliquid account access from the real trading environment.
- Confirm exact instrument IDs, lot size, min notional, and leverage limits.
- Confirm maker/taker fees and whether taker entry still leaves positive 8h net.
- Confirm funding timestamp alignment on both venues.
- Confirm that mark/index basis does not dominate expected funding capture.
- Confirm margin, liquidation buffer, and collateral transfer path.
- Define exit if net proxy turns negative or either leg cannot be adjusted.

## Why This Candidate

This candidate survived the short persistence probe with positive 8h net proxy in every snapshot. That does not prove a real edge; it only makes it the first candidate worth converting from screen output into a venue-specific paper workflow.
