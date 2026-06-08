# Current Protocol Fee Execution Context

This joins protocol fee-growth lag candidates to current perp venue coverage and Hyperliquid public-book context. It is a paper-observation gate, not a live trade instruction.

| token | protocol | score | price7d | venues | HL funding | HL volume 24h | spread bps | depth 10bps USD | action | next step |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| AAVE | Aave V3 | 87.9820 | -20.09 | 2 | 0.0197 | 4722626 | 2.4959 | 35915 | paper_observation_ready | paper-label AAVE fee-growth lag with 4h/12h/24h return, funding, spread, and depth costs |
| CRV | Curve DEX | 59.0582 | -6.71 | 2 | 0.1095 | 2180617 | 0.5009 | 23800 | paper_observation_ready | paper-label CRV fee-growth lag with 4h/12h/24h return, funding, spread, and depth costs |
| MORPHO | Morpho Blue | 54.8658 | -12.57 | 2 | 0.1095 | 3139663 | 3.8232 | 6508 | paper_observation_ready | paper-label MORPHO fee-growth lag with 4h/12h/24h return, funding, spread, and depth costs |
| JUP | Jupiter Perpetual Exchange | 95.9652 | -18.67 | 2 | 0.1095 | 1826139 | 3.6776 | 4572 | thin_volume_watch | keep JUP as a low-liquidity paper label, not an execution candidate |
| UNI | Uniswap V3 | 78.3056 | -12.95 | 2 | 0.1095 | 1415875 | 1.9296 | 19268 | thin_volume_watch | keep UNI as a low-liquidity paper label, not an execution candidate |
| PENDLE | Pendle | 42.9349 | -8.44 | 2 | 0.0837 | 1240164 | 6.2622 | 5136 | thin_volume_watch | keep PENDLE as a low-liquidity paper label, not an execution candidate |

## Interpretation

`paper_observation_ready` only means the fee-growth lag thesis has current venue coverage and public-book context that is not obviously blocking a small paper observation. It does not prove alpha, fill quality, borrow availability, account fees, or liquidation safety.
