# OKX-Hyperliquid Order Constraints

Generated: `2026-06-08T00:31:02.820019+00:00`

This is not an order instruction. It is a paper order-shape check.

## Candidate

- Asset: `MON`
- Paper notional: `1000` USDT
- Long venue: `OkxSwap`
- Short venue: `HlPerp`

## OKX Leg

- Instrument: `MON-USDT-SWAP`
- Raw contracts: `4457.321149988856697125027858`
- Rounded contracts: `4457`
- Rounded notional: `999.92795` USDT
- Min size: `1`
- Lot size: `1`
- Tick size: `0.00001`
- Max leverage: `50`
- Size valid: `True`

## Hyperliquid Leg

- Raw size: `44573.21149988856697125027858` MON
- Rounded size: `44573` MON
- Rounded notional: `999.995255` USDT
- Size decimals: `0`
- Max leverage: `5`
- Day notional volume: `4107191.5541219986` USDT
- Size valid: `True`

## Still Unknown

- Actual account access and jurisdiction from the trading environment.
- Actual maker/taker fee tier on both venues.
- Whether maker execution is realistic without losing the funding window.
- Margin mode, collateral movement, liquidation buffer, and kill switch.
- Whether the funding spread persists at order-entry time.

## Notes

Public instrument constraints allow the paper size shape
