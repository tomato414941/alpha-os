# OKX-Hyperliquid Fee Ceiling

This estimates the maximum equal per-fill fee bps each venue can charge before the event-window edge is erased. It uses the execution-mode slippage already measured from the public book.

| asset | mode | max fee 8h bps/fill/venue | max fee 24h bps/fill/venue | both touch | OKX only | HL only | capacity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BABY | both_maker | 0.92155 | 2.764675 | 0 | 0.2 | 0.2 | 18182.63949194 |
| ZEC | both_maker | 0.602075 | 1.806225 | 0 | 0.6 | 0.2 | 106210.05564167 |
| ZEC | okx_cross_hl_maker | 0.5405 | 1.74465 | 0 | 0.6 | 0.2 | 106210.05564167 |
| ZEC | okx_maker_hl_cross | 0.355375 | 1.559525 | 0 | 0.6 | 0.2 | 106210.05564167 |
| ZEC | both_cross | 0.2938 | 1.49795 | 0 | 0.6 | 0.2 | 106210.05564167 |
| BTC | both_maker | 0.26455 | 0.793675 | 0.2 | 0.4 | 0.2 | 422448.80855333 |
| BTC | okx_cross_hl_maker | 0.2605 | 0.789625 | 0.2 | 0.4 | 0.2 | 422448.80855333 |
| BTC | okx_maker_hl_cross | 0.22405 | 0.753175 | 0.2 | 0.4 | 0.2 | 422448.80855333 |
| BTC | both_cross | 0.22 | 0.749125 | 0.2 | 0.4 | 0.2 | 422448.80855333 |
| JTO | both_maker | 0.146925 | 0.440775 | 0 | 0.8 | 0.2 | 54543.56750198 |
| JTO | okx_cross_hl_maker | -1.261125 | -0.967275 | 0 | 0.8 | 0.2 | 54543.56750198 |
| JTO | okx_maker_hl_cross | -1.46125 | -1.1674 | 0 | 0.8 | 0.2 | 54543.56750198 |
| BABY | okx_maker_hl_cross | -3.1513 | -1.308175 | 0 | 0.2 | 0.2 | 18182.63949194 |
| BABY | okx_cross_hl_maker | -3.846875 | -2.00375 | 0 | 0.2 | 0.2 | 18182.63949194 |
| JTO | both_cross | -2.8693 | -2.57545 | 0 | 0.8 | 0.2 | 54543.56750198 |
| BABY | both_cross | -7.919725 | -6.0766 | 0 | 0.2 | 0.2 | 18182.63949194 |

## Interpretation

Negative fee ceilings mean the slippage-adjusted edge is already gone before fees. A ceiling below the actual account fee means the mode should not be promoted even if the raw funding spread looks positive.

- BABY has the largest maker-only ceiling in this snapshot, but it has no same-window two-leg maker touch in this sample, low capacity, and loses most of the edge when one leg crosses.
- ZEC has the best current one-leg-cross 24h headroom, especially `okx_cross_hl_maker`, but it is still capacity and stability constrained.
- BTC has the cleanest capacity and survives all execution modes under very-low fees, but its 8h edge is extremely fee-sensitive.
- JTO is weak in this event-window snapshot: even maker-only 24h has only a small fee ceiling, and one-leg-cross modes are already negative.
- The next hard gate is the real account fee tier. Without that, raw funding spread is not enough to promote a mode.
