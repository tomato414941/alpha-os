# Current Protocol Fee Price Context

This joins protocol fee-growth valuation to current CoinGecko price movement. It looks for fee growth that is not yet fully chased by price, and for crowded price-confirmed setups.

| token | protocol | status | score | fee/mcap | fee/fdv | fee growth 7d | price 24h | price 7d | price 30d | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| JUP | Jupiter Perpetual Exchange | fee_growth_price_lag_candidate | 94.7857 | 0.5802 | 0.2807 | 232.55 | 3.76 | -13.46 | -36.48 | paper-label JUP as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| AAVE | Aave V3 | fee_growth_price_lag_candidate | 87.9820 | 0.9674 | 0.9178 | 129.82 | 2.44 | -19.45 | -31.99 | paper-label AAVE as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| UNI | Uniswap V3 | fee_growth_price_lag_candidate | 77.9671 | 0.3298 | 0.2295 | 126.91 | 1.64 | -12.33 | -29.72 | paper-label UNI as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| ENA | Ethena USDe | fee_price_context_watch | 60.2773 | 0.4443 | 0.2753 | 27.49 | -3.71 | 2.86 | -32.51 | collect another ENA fee and price snapshot before promotion |
| CRV | Curve DEX | fee_growth_price_lag_candidate | 57.0025 | 0.1831 | 0.1168 | 222.41 | 3.90 | -4.77 | -18.84 | paper-label CRV as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| AERO | Aerodrome Slipstream | fee_price_context_watch | 54.2240 | 0.3401 | 0.1685 | 73.72 | 3.80 | -12.54 | -30.54 | collect another AERO fee and price snapshot before promotion |
| MORPHO | Morpho Blue | fee_growth_price_lag_candidate | 44.0323 | 0.1628 | 0.1052 | 140.49 | 13.02 | -3.19 | -8.93 | paper-label MORPHO as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| PENDLE | Pendle | fee_growth_price_lag_candidate | 38.0414 | 0.1150 | 0.0697 | 158.66 | 3.28 | -3.71 | -35.92 | paper-label PENDLE as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| UNI | Uniswap V4 | fee_decay_price_weakness_context | 32.2231 | 0.1503 | 0.1046 | -32.64 | 1.64 | -12.33 | -29.72 | test whether UNI fee decay and weak price persist before any short thesis |
| HYPE | Hyperliquid Perps | fee_price_context_watch | 19.6495 | 0.0690 | 0.0161 | 111.41 | 9.98 | -10.48 | 48.21 | collect another HYPE fee and price snapshot before promotion |
| SOL | Solana | fee_decay_price_weakness_context | 10.7135 | 0.0080 | 0.0074 | -8.33 | 3.58 | -15.86 | -28.07 | test whether SOL fee decay and weak price persist before any short thesis |
| ETH | Ethereum | fee_price_context_watch | 8.1194 | 0.0015 | 0.0015 | 78.11 | 4.13 | -13.94 | -26.90 | collect another ETH fee and price snapshot before promotion |
| HYPE | Hyper Foundation HYPE Staking | fee_price_context_watch | 2.4466 | 0.0181 | 0.0042 | 2.11 | 9.98 | -10.48 | 48.21 | collect another HYPE fee and price snapshot before promotion |

## Interpretation

`fee_growth_price_lag_candidate` is the most interesting long setup class here: fees are strong, but the token has not obviously chased over the last week. `fee_growth_price_chase_risk` may still work, but it needs stricter entry timing and drawdown control.
