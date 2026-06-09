# Current Protocol Fee Execution Context

This joins protocol fee-growth lag candidates to current perp venue coverage and Hyperliquid public-book context. It is a paper-observation gate, not a live trade instruction.

| token | protocol | score | price7d | venues | HL funding | HL volume 24h | spread bps | depth 10bps USD | action | next step |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| AAVE | Aave V3 | 87.9820 | -20.09 | 2 | 0.0912 | 3869137 | 1.1041 | 41673 | paper_observation_ready | paper-label AAVE fee-growth lag with 4h/12h/24h return, funding, spread, and depth costs |
| CRV | Curve DEX | 59.0582 | -6.71 | 2 | 0.1095 | 2101471 | 4.5341 | 24655 | paper_observation_ready | paper-label CRV fee-growth lag with 4h/12h/24h return, funding, spread, and depth costs |
| MORPHO | Morpho Blue | 54.8658 | -12.57 | 2 | 0.1095 | 3377551 | 2.2182 | 5651 | paper_observation_ready | paper-label MORPHO fee-growth lag with 4h/12h/24h return, funding, spread, and depth costs |
| JUP | Jupiter Perpetual Exchange | 95.9652 | -18.67 | 2 | -0.1616 | 1848103 | 3.1459 | 4068 | thin_volume_watch | keep JUP as a low-liquidity paper label, not an execution candidate |
| UNI | Uniswap V3 | 78.3056 | -12.95 | 2 | 0.0989 | 1282854 | 1.1951 | 21914 | thin_volume_watch | keep UNI as a low-liquidity paper label, not an execution candidate |
| PENDLE | Pendle | 42.9349 | -8.44 | 2 | 0.1095 | 1071314 | 4.7125 | 6181 | thin_volume_watch | keep PENDLE as a low-liquidity paper label, not an execution candidate |

## Interpretation

`paper_observation_ready` only means the fee-growth lag thesis has current venue coverage and public-book context that is not obviously blocking a small paper observation. It does not prove alpha, fill quality, borrow availability, account fees, or liquidation safety.
