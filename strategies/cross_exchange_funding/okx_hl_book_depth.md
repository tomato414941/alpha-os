# OKX-Hyperliquid Book Depth

Generated: `2026-06-07T14:32:42.461077+00:00`

This is not an order instruction. It checks taker depth for the paper size.

| venue | side | target notional | top level notional | avg fill | slippage bps | levels | full |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| OkxSwap | buy | 1000 | 104.935 | 0.03386587 | 6.16586322 | 3 | True |
| HlPerp | sell | 1000 | 246.145465 | 0.03388013 | 5.12816838 | 3 | True |

- Combined taker slippage bps: `11.2940316`
- Notes: paper size fills, but consumes multiple visible levels

## Interpretation

This check only measures visible public book depth. It does not prove maker fill probability, account fees, post-only behavior, or whether the funding spread persists during execution.
