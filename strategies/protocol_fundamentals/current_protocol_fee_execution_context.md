# Current Protocol Fee Execution Context

This joins protocol fee-growth lag candidates to current perp venue coverage and Hyperliquid public-book context. It is a paper-observation gate, not a live trade instruction.

| token | protocol | score | price7d | venues | HL funding | HL volume 24h | spread bps | depth 10bps USD | action | next step |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| AAVE | Aave V3 | 87.9820 | -19.45 | 2 | 0.0664 | 5130762 | 4.7969 | 34036 | paper_observation_ready | paper-label AAVE fee-growth lag with 4h/12h/24h return, funding, spread, and depth costs |
| CRV | Curve DEX | 57.0025 | -4.77 | 2 | 0.1095 | 2250768 | 3.9744 | 21208 | paper_observation_ready | paper-label CRV fee-growth lag with 4h/12h/24h return, funding, spread, and depth costs |
| JUP | Jupiter Perpetual Exchange | 94.7857 | -13.46 | 2 | -0.2622 | 1728507 | 4.3265 | 6281 | thin_volume_watch | keep JUP as a low-liquidity paper label, not an execution candidate |
| UNI | Uniswap V3 | 77.9671 | -12.33 | 2 | 0.1095 | 1534329 | 4.2851 | 19929 | thin_volume_watch | keep UNI as a low-liquidity paper label, not an execution candidate |
| MORPHO | Morpho Blue | 44.0323 | -3.19 | 2 | 0.1095 | 1898682 | 8.3238 | 2912 | thin_volume_watch | keep MORPHO as a low-liquidity paper label, not an execution candidate |
| PENDLE | Pendle | 38.0414 | -3.71 | 2 | 0.1095 | 1178467 | 3.8719 | 3724 | thin_volume_watch | keep PENDLE as a low-liquidity paper label, not an execution candidate |

## Interpretation

`paper_observation_ready` only means the fee-growth lag thesis has current venue coverage and public-book context that is not obviously blocking a small paper observation. It does not prove alpha, fill quality, borrow availability, account fees, or liquidation safety.
