# OKX-Hyperliquid Order Constraints

Generated: `2026-06-07T14:32:42.335984+00:00`

This is not an order instruction. It is a paper order-shape check.

## Candidate

- Asset: `STABLE`
- Paper notional: `1000` USDT
- Long venue: `OkxSwap`
- Short venue: `HlPerp`

## OKX Leg

- Instrument: `STABLE-USDT-SWAP`
- Raw contracts: `295.0113579372805853025341476`
- Rounded contracts: `295`
- Rounded notional: `999.9615` USDT
- Min size: `1`
- Lot size: `1`
- Tick size: `0.00001`
- Max leverage: `20`
- Size valid: `True`

## Hyperliquid Leg

- Raw size: `29501.13579372805853025341476` STABLE
- Rounded size: `29501` STABLE
- Rounded notional: `999.995397` USDT
- Size decimals: `0`
- Max leverage: `3`
- Day notional volume: `1183700.5671269991` USDT
- Size valid: `True`

## Still Unknown

- Actual account access and jurisdiction from the trading environment.
- Actual maker/taker fee tier on both venues.
- Whether maker execution is realistic without losing the funding window.
- Margin mode, collateral movement, liquidation buffer, and kill switch.
- Whether the funding spread persists at order-entry time.

## Notes

Public instrument constraints allow the paper size shape
