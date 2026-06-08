# Paper Trade Ticket

Generated: 2026-06-08T00:08:09.999526+00:00

This is not a trade instruction. It is an operational feasibility ticket.

## Candidate

- Asset: `STABLE`
- Long venue: `BybitPerp`
- Short venue: `HlPerp`
- Annualized spread snapshot: `2.51400524`
- Hyperliquid 24h notional volume: `1176450.45`
- Hyperliquid impact spread: `0.00298578`
- Source timestamp: `2026-06-07T14:17:54.065127+00:00`
- Notes: Hyperliquid context available

## Required Checks Before Any Real Order

- Confirm both venues are accessible from the actual account and jurisdiction.
- Confirm symbol availability, lot size, min notional, and leverage limits.
- Confirm maker/taker fees and whether the spread survives taker execution.
- Confirm borrow, margin, liquidation buffer, and funding interval timing.
- Confirm depth for the intended notional on both legs.
- Confirm that predicted funding still exists immediately before entry.
- Define exit condition, max loss, and kill switch.

## First Falsification

If this ticket cannot be converted into executable venue-specific order details with fees, size, and risk limits, this lane is not operational yet.
