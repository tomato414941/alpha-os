# OKX-Hyperliquid Book Depth

Generated: `2026-06-08T00:31:03.805442+00:00`

This is not an order instruction. It checks taker depth for the paper size.

| venue | side | target notional | top level notional | avg fill | slippage bps | levels | full |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| OkxSwap | buy | 1000 | 424.65 | 0.02235575 | 4.81199689 | 2 | True |
| HlPerp | sell | 1000 | 250.095688 | 0.0224035 | 4.46346646 | 2 | True |

- Combined taker slippage bps: `9.27546334`
- Notes: paper size fills, but consumes multiple visible levels

## Interpretation

This check only measures visible public book depth. It does not prove maker fill probability, account fees, post-only behavior, or whether the funding spread persists during execution.
