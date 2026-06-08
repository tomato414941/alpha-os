# Current Protocol Fee Execution Context

This joins protocol fee-growth lag candidates to current perp venue coverage and Hyperliquid public-book context. It is a paper-observation gate, not a live trade instruction.

| token | protocol | score | price7d | venues | HL funding | HL volume 24h | spread bps | depth 10bps USD | action | next step |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| AAVE | Aave V3 | 87.9820 | -21.16 | 2 | 0.1095 | 4849988 | 3.7375 | 36369 | paper_observation_ready | paper-label AAVE fee-growth lag with 4h/12h/24h return, funding, spread, and depth costs |
| CRV | Curve DEX | 57.6620 | -5.51 | 2 | 0.1095 | 2415462 | 1.9812 | 40183 | paper_observation_ready | paper-label CRV fee-growth lag with 4h/12h/24h return, funding, spread, and depth costs |
| JUP | Jupiter Perpetual Exchange | 96.7959 | -18.29 | 2 | -0.6235 | 1493068 | 4.3812 | 3534 | thin_volume_watch | keep JUP as a low-liquidity paper label, not an execution candidate |
| UNI | Uniswap V3 | 81.1041 | -15.24 | 2 | 0.1095 | 1385347 | 2.3699 | 19218 | thin_volume_watch | keep UNI as a low-liquidity paper label, not an execution candidate |
| MORPHO | Morpho Blue | 51.8802 | -9.16 | 2 | 0.1095 | 1243717 | 5.4924 | 525 | thin_volume_watch | keep MORPHO as a low-liquidity paper label, not an execution candidate |
| PENDLE | Pendle | 39.4083 | -4.91 | 2 | 0.1095 | 1065284 | 3.8961 | 3314 | thin_volume_watch | keep PENDLE as a low-liquidity paper label, not an execution candidate |

## Interpretation

`paper_observation_ready` only means the fee-growth lag thesis has current venue coverage and public-book context that is not obviously blocking a small paper observation. It does not prove alpha, fill quality, borrow availability, account fees, or liquidation safety.
