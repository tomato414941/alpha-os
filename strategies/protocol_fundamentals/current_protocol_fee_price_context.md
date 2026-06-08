# Current Protocol Fee Price Context

This joins protocol fee-growth valuation to current CoinGecko price movement. It looks for fee growth that is not yet fully chased by price, and for crowded price-confirmed setups.

| token | protocol | status | score | fee/mcap | fee/fdv | fee growth 7d | price 24h | price 7d | price 30d | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| JUP | Jupiter Perpetual Exchange | fee_growth_price_lag_candidate | 95.9652 | 0.5727 | 0.2771 | 232.55 | 8.34 | -18.67 | -35.83 | paper-label JUP as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| AAVE | Aave V3 | fee_growth_price_lag_candidate | 87.9820 | 0.9727 | 0.9229 | 129.82 | 4.50 | -20.09 | -33.09 | paper-label AAVE as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| UNI | Uniswap V3 | fee_growth_price_lag_candidate | 78.3056 | 0.3256 | 0.2266 | 126.91 | 4.63 | -12.95 | -30.06 | paper-label UNI as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| ENA | Ethena USDe | fee_price_context_watch | 59.8599 | 0.4376 | 0.2711 | 27.49 | 3.18 | 1.14 | -31.04 | collect another ENA fee and price snapshot before promotion |
| CRV | Curve DEX | fee_growth_price_lag_candidate | 59.0582 | 0.1838 | 0.1172 | 222.41 | 5.19 | -6.71 | -20.94 | paper-label CRV as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| MORPHO | Morpho Blue | fee_growth_price_lag_candidate | 54.8658 | 0.1716 | 0.1109 | 140.49 | 10.03 | -12.57 | -13.19 | paper-label MORPHO as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| AERO | Aerodrome Slipstream | fee_price_context_watch | 54.0679 | 0.3369 | 0.1670 | 73.72 | 6.57 | -14.25 | -30.21 | collect another AERO fee and price snapshot before promotion |
| PENDLE | Pendle | fee_growth_price_lag_candidate | 42.9349 | 0.1160 | 0.0703 | 158.66 | 4.58 | -8.44 | -37.29 | paper-label PENDLE as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| UNI | Uniswap V4 | fee_decay_price_weakness_context | 31.9037 | 0.1484 | 0.1033 | -32.64 | 4.63 | -12.95 | -30.06 | test whether UNI fee decay and weak price persist before any short thesis |
| HYPE | Hyperliquid Perps | fee_price_context_watch | 19.7990 | 0.0702 | 0.0164 | 111.41 | 10.68 | -12.80 | 47.74 | collect another HYPE fee and price snapshot before promotion |
| SOL | Solana | fee_decay_price_weakness_context | 10.7049 | 0.0080 | 0.0074 | -8.33 | 5.55 | -16.58 | -27.58 | test whether SOL fee decay and weak price persist before any short thesis |
| ETH | Ethereum | fee_price_context_watch | 8.1198 | 0.0015 | 0.0015 | 78.11 | 4.56 | -15.14 | -27.53 | collect another ETH fee and price snapshot before promotion |
| HYPE | Hyper Foundation HYPE Staking | fee_price_context_watch | 2.4859 | 0.0185 | 0.0043 | 2.11 | 10.68 | -12.80 | 47.74 | collect another HYPE fee and price snapshot before promotion |

## Interpretation

`fee_growth_price_lag_candidate` is the most interesting long setup class here: fees are strong, but the token has not obviously chased over the last week. `fee_growth_price_chase_risk` may still work, but it needs stricter entry timing and drawdown control.
