# OKX-Hyperliquid Book Depth

Generated: `2026-06-07T12:11:18.145192+00:00`

This is not an order instruction. It checks taker depth for the paper size.

| venue | side | target notional | top level notional | avg fill | slippage bps | levels | full |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| OkxSwap | buy | 995.58645 | 275099.24052 | 62279.1 | 0.00802838 | 1 | True |
| HlPerp | sell | 999.969535 | 288224.46108 | 62292 | 0.08026713 | 1 | True |

- Combined taker slippage bps: `0.08829551`
- Notes: paper size fits inside the top visible level on both venues

## Interpretation

This check only measures visible public book depth. It does not prove maker fill probability, account fees, post-only behavior, or whether the funding spread persists during execution.
