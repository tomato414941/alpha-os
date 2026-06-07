# OKX-Hyperliquid Funding Alignment

Generated: `2026-06-07T12:04:00.737526+00:00`

This is not a trade instruction. It checks funding timestamp alignment.

## Candidate

- Asset: `BTC`
- Long venue: `OkxSwap`
- Short venue: `HlPerp`

## Funding Times

- OKX instrument: `BTC-USDT-SWAP`
- OKX current funding rate: `-0.000017878722904`
- OKX long expected rate per event: `0.000017878722904`
- OKX first funding time: `2026-06-07T16:00:00+00:00`
- OKX interval hours: `8`
- Hyperliquid funding rate: `0.0000125`
- Hyperliquid short expected rate per event: `0.0000125`
- Hyperliquid first funding time: `2026-06-07T13:00:00+00:00`
- Hyperliquid interval hours: `1`
- First event gap hours: `3`

## Event Counts

- OKX events within 8h: `1`
- Hyperliquid events within 8h: `8`
- OKX events within 24h: `3`
- Hyperliquid events within 24h: `24`

## Notes

Current signs match the paper direction: long OKX and short Hyperliquid both expect funding income

## Still Unknown

- Whether these rates persist until each funding event.
- Whether entry can be completed before the relevant funding windows.
- Exact account fee tier and collateral/margin state.
