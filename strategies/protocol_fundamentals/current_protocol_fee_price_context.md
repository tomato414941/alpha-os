# Current Protocol Fee Price Context

This joins protocol fee-growth valuation to current CoinGecko price movement. It looks for fee growth that is not yet fully chased by price, and for crowded price-confirmed setups.

| token | protocol | status | score | fee/mcap | fee/fdv | fee growth 7d | price 24h | price 7d | price 30d | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| JUP | Jupiter Perpetual Exchange | fee_growth_price_lag_candidate | 94.3453 | 0.6135 | 0.2969 | 196.60 | 3.58 | -16.31 | -35.47 | paper-label JUP as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| AAVE | Aave V3 | fee_growth_price_lag_candidate | 87.8950 | 1.0017 | 0.9503 | 128.95 | 6.58 | -21.37 | -33.30 | paper-label AAVE as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| UNI | Uniswap V3 | fee_growth_price_lag_candidate | 77.8378 | 0.3375 | 0.2348 | 95.01 | 5.38 | -14.85 | -30.28 | paper-label UNI as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| MORPHO | Morpho Blue | fee_growth_price_lag_candidate | 60.8602 | 0.1865 | 0.1205 | 151.57 | 5.34 | -19.58 | -18.63 | paper-label MORPHO as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| ENA | Ethena USDe | fee_price_context_watch | 60.2752 | 0.4446 | 0.2755 | 27.29 | 1.10 | 3.97 | -32.26 | collect another ENA fee and price snapshot before promotion |
| CRV | Curve DEX | fee_growth_price_lag_candidate | 59.0203 | 0.1902 | 0.1212 | 190.42 | 7.94 | -8.84 | -23.91 | paper-label CRV as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| AERO | Aerodrome Slipstream | fee_price_context_watch | 50.5962 | 0.3573 | 0.1771 | 28.89 | 6.58 | -18.20 | -25.86 | collect another AERO fee and price snapshot before promotion |
| PENDLE | Pendle | fee_growth_price_lag_candidate | 39.1458 | 0.1204 | 0.0730 | 118.68 | 4.15 | -7.94 | -39.17 | paper-label PENDLE as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| UNI | Uniswap V4 | fee_decay_price_weakness_context | 32.1903 | 0.1535 | 0.1068 | -38.41 | 5.38 | -14.85 | -30.28 | test whether UNI fee decay and weak price persist before any short thesis |
| HYPE | Hyperliquid Perps | fee_price_context_watch | 19.3177 | 0.0762 | 0.0177 | 99.26 | 5.10 | -15.12 | 37.36 | collect another HYPE fee and price snapshot before promotion |
| SOL | Solana | fee_decay_price_weakness_context | 10.7636 | 0.0084 | 0.0077 | -8.44 | 7.54 | -19.03 | -28.15 | test whether SOL fee decay and weak price persist before any short thesis |
| ETH | Ethereum | fee_price_context_watch | 7.9236 | 0.0016 | 0.0016 | 76.03 | 7.70 | -16.45 | -27.40 | collect another ETH fee and price snapshot before promotion |
| HYPE | Hyper Foundation HYPE Staking | fee_price_context_watch | 3.2891 | 0.0200 | 0.0047 | 8.24 | 5.10 | -15.12 | 37.36 | collect another HYPE fee and price snapshot before promotion |

## Interpretation

`fee_growth_price_lag_candidate` is the most interesting long setup class here: fees are strong, but the token has not obviously chased over the last week. `fee_growth_price_chase_risk` may still work, but it needs stricter entry timing and drawdown control.
