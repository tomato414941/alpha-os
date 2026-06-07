# OKX-Hyperliquid Order Constraints

Generated: `2026-06-07T12:00:03.921873+00:00`

This is not an order instruction. It is a paper order-shape check.

## Candidate

- Asset: `BTC`
- Paper notional: `1000` USDT
- Long venue: `OkxSwap`
- Short venue: `HlPerp`

## OKX Leg

- Instrument: `BTC-USDT-SWAP`
- Raw contracts: `1.597048654087246767972786291`
- Rounded contracts: `1.59`
- Rounded notional: `995.58645` USDT
- Min size: `0.01`
- Lot size: `0.01`
- Tick size: `0.1`
- Max leverage: `100`
- Size valid: `True`

## Hyperliquid Leg

- Raw size: `0.01597048654087246767972786291` BTC
- Rounded size: `0.01597` BTC
- Rounded notional: `999.969535` USDT
- Size decimals: `5`
- Max leverage: `40`
- Day notional volume: `2233407017.6160302162` USDT
- Size valid: `True`

## Still Unknown

- Actual account access and jurisdiction from the trading environment.
- Actual maker/taker fee tier on both venues.
- Whether maker execution is realistic without losing the funding window.
- Margin mode, collateral movement, liquidation buffer, and kill switch.
- Whether the funding spread persists at order-entry time.

## Notes

Public instrument constraints allow the paper size shape
