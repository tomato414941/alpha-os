# Current Protocol Fee Price Context

This joins protocol fee-growth valuation to current CoinGecko price movement. It looks for fee growth that is not yet fully chased by price, and for crowded price-confirmed setups.

| token | protocol | status | score | fee/mcap | fee/fdv | fee growth 7d | price 24h | price 7d | price 30d | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| JUP | Jupiter Perpetual Exchange | fee_growth_price_lag_candidate | 96.8203 | 0.5904 | 0.2857 | 232.55 | 2.50 | -18.75 | -35.53 | paper-label JUP as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| AAVE | Aave V3 | fee_growth_price_lag_candidate | 87.9820 | 0.9795 | 0.9293 | 129.82 | 2.19 | -21.44 | -33.55 | paper-label AAVE as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| UNI | Uniswap V3 | fee_growth_price_lag_candidate | 81.1328 | 0.3369 | 0.2344 | 126.91 | -0.46 | -15.74 | -31.59 | paper-label UNI as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| ENA | Ethena USDe | fee_price_context_watch | 60.5170 | 0.4482 | 0.2777 | 27.49 | -3.18 | 0.67 | -32.54 | collect another ENA fee and price snapshot before promotion |
| CRV | Curve DEX | fee_growth_price_lag_candidate | 58.3322 | 0.1835 | 0.1170 | 222.41 | 4.71 | -6.05 | -20.06 | paper-label CRV as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| AERO | Aerodrome Slipstream | fee_price_context_watch | 54.4828 | 0.3452 | 0.1711 | 73.72 | 2.75 | -16.60 | -30.12 | collect another AERO fee and price snapshot before promotion |
| MORPHO | Morpho Blue | fee_growth_price_lag_candidate | 52.5080 | 0.1752 | 0.1132 | 140.49 | 4.86 | -9.62 | -14.62 | paper-label MORPHO as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| PENDLE | Pendle | fee_growth_price_lag_candidate | 39.5859 | 0.1161 | 0.0704 | 158.66 | 3.60 | -5.08 | -37.59 | paper-label PENDLE as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| UNI | Uniswap V4 | fee_decay_price_weakness_context | 32.7714 | 0.1535 | 0.1068 | -32.64 | -0.46 | -15.74 | -31.59 | test whether UNI fee decay and weak price persist before any short thesis |
| HYPE | Hyperliquid Perps | fee_price_context_watch | 20.0889 | 0.0726 | 0.0169 | 111.41 | 6.43 | -16.37 | 40.35 | collect another HYPE fee and price snapshot before promotion |
| SOL | Solana | fee_decay_price_weakness_context | 10.7344 | 0.0082 | 0.0075 | -8.33 | 3.23 | -18.36 | -29.32 | test whether SOL fee decay and weak price persist before any short thesis |
| ETH | Ethereum | fee_price_context_watch | 8.1224 | 0.0016 | 0.0016 | 78.11 | 3.83 | -15.53 | -27.74 | collect another ETH fee and price snapshot before promotion |
| HYPE | Hyper Foundation HYPE Staking | fee_price_context_watch | 2.5621 | 0.0191 | 0.0044 | 2.11 | 6.43 | -16.37 | 40.35 | collect another HYPE fee and price snapshot before promotion |

## Interpretation

`fee_growth_price_lag_candidate` is the most interesting long setup class here: fees are strong, but the token has not obviously chased over the last week. `fee_growth_price_chase_risk` may still work, but it needs stricter entry timing and drawdown control.
