# Current Protocol Fee Execution Context

This joins protocol fee-growth lag candidates to current perp venue coverage and Hyperliquid public-book context. It is a paper-observation gate, not a live trade instruction.

| token | protocol | score | price7d | venues | HL funding | HL volume 24h | spread bps | depth 10bps USD | action | next step |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| AAVE | Aave V3 | 87.9800 | -21.82 | 2 | 0.1095 | 6734140 | 2.3353 | 25720 | paper_observation_ready | paper-label AAVE fee-growth lag with 4h/12h/24h return, funding, spread, and depth costs |
| CRV | Curve DEX | 59.9474 | -8.57 | 2 | 0.1095 | 2731617 | 8.5507 | 19621 | wide_spread_watch | wait for tighter CRV spread or use another venue before paper observation |
| JUP | Jupiter Perpetual Exchange | 93.9263 | -16.78 | 2 | 0.0073 | 1325756 | 1.8771 | 2542 | thin_volume_watch | keep JUP as a low-liquidity paper label, not an execution candidate |
| UNI | Uniswap V3 | 77.8233 | -14.58 | 2 | 0.1095 | 1401861 | 3.4860 | 20218 | thin_volume_watch | keep UNI as a low-liquidity paper label, not an execution candidate |
| MORPHO | Morpho Blue | 58.5616 | -19.06 | 2 | 0.1095 | 910841 | 7.4390 | 740 | thin_volume_watch | keep MORPHO as a low-liquidity paper label, not an execution candidate |
| PENDLE | Pendle | 40.2142 | -5.79 | 2 | 0.0133 | 1109433 | 3.1097 | 4968 | thin_volume_watch | keep PENDLE as a low-liquidity paper label, not an execution candidate |

## Interpretation

`paper_observation_ready` only means the fee-growth lag thesis has current venue coverage and public-book context that is not obviously blocking a small paper observation. It does not prove alpha, fill quality, borrow availability, account fees, or liquidation safety.
