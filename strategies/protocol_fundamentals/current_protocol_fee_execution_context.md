# Current Protocol Fee Execution Context

This joins protocol fee-growth lag candidates to current perp venue coverage and Hyperliquid public-book context. It is a paper-observation gate, not a live trade instruction.

| token | protocol | score | price7d | venues | HL funding | HL volume 24h | spread bps | depth 10bps USD | action | next step |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| AAVE | Aave V3 | 87.9820 | -18.98 | 2 | 0.0643 | 5126004 | 2.9267 | 47586 | paper_observation_ready | paper-label AAVE fee-growth lag with 4h/12h/24h return, funding, spread, and depth costs |
| CRV | Curve DEX | 56.5404 | -4.39 | 2 | 0.1095 | 2249135 | 2.9729 | 21315 | paper_observation_ready | paper-label CRV fee-growth lag with 4h/12h/24h return, funding, spread, and depth costs |
| JUP | Jupiter Perpetual Exchange | 94.6419 | -13.34 | 2 | -0.1792 | 1743194 | 3.1076 | 3592 | thin_volume_watch | keep JUP as a low-liquidity paper label, not an execution candidate |
| UNI | Uniswap V3 | 77.7421 | -12.13 | 2 | 0.1095 | 1530905 | 2.3329 | 11637 | thin_volume_watch | keep UNI as a low-liquidity paper label, not an execution candidate |
| MORPHO | Morpho Blue | 44.6764 | -3.75 | 2 | 0.1095 | 1982741 | 10.1201 | 3698 | thin_volume_watch | keep MORPHO as a low-liquidity paper label, not an execution candidate |
| PENDLE | Pendle | 37.6100 | -3.32 | 2 | 0.1095 | 1178565 | 6.9587 | 4096 | thin_volume_watch | keep PENDLE as a low-liquidity paper label, not an execution candidate |

## Interpretation

`paper_observation_ready` only means the fee-growth lag thesis has current venue coverage and public-book context that is not obviously blocking a small paper observation. It does not prove alpha, fill quality, borrow availability, account fees, or liquidation safety.
