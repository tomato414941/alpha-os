# OKX-Hyperliquid Fee Ceiling

This estimates the maximum equal per-fill fee bps each venue can charge before the event-window edge is erased. It uses the execution-mode slippage already measured from the public book.

| asset | mode | max fee 8h bps/fill/venue | max fee 24h bps/fill/venue | both touch | OKX only | HL only | capacity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ZEC | both_maker | 0.602075 | 1.806225 | 0.2 | 0.6 | 0.2 | 106210.05564167 |
| ZEC | okx_cross_hl_maker | 0.540375 | 1.744525 | 0.2 | 0.6 | 0.2 | 106210.05564167 |
| ZEC | okx_maker_hl_cross | 0.108525 | 1.312675 | 0.2 | 0.6 | 0.2 | 106210.05564167 |
| ZEC | both_cross | 0.0468 | 1.25095 | 0.2 | 0.6 | 0.2 | 106210.05564167 |
| BTC | both_maker | 0.26455 | 0.793675 | 0 | 0.6 | 0.2 | 422448.80855333 |
| BTC | okx_cross_hl_maker | 0.2605 | 0.789625 | 0 | 0.6 | 0.2 | 422448.80855333 |
| BTC | okx_maker_hl_cross | 0.224075 | 0.7532 | 0 | 0.6 | 0.2 | 422448.80855333 |
| BTC | both_cross | 0.220025 | 0.74915 | 0 | 0.6 | 0.2 | 422448.80855333 |

## Interpretation

Negative fee ceilings mean the slippage-adjusted edge is already gone before fees. A ceiling below the actual account fee means the mode should not be promoted even if the raw funding spread looks positive.

- BTC 8h is extremely fee-sensitive: it needs roughly 0.26 bps or less per fill per venue even before queue-position and adverse-selection risk.
- BTC 24h has more room at roughly 0.79 bps per fill per venue, but its funding event cadence is slower.
- ZEC 24h has the largest fee headroom in this snapshot, including `okx_cross_hl_maker`, but its capacity and event stability are weaker than BTC.
- The next hard gate is the real account fee tier. Without that, raw funding spread is not enough to promote a mode.
