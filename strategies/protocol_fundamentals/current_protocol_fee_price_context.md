# Current Protocol Fee Price Context

This joins protocol fee-growth valuation to current CoinGecko price movement. It looks for fee growth that is not yet fully chased by price, and for crowded price-confirmed setups.

| token | protocol | status | score | fee/mcap | fee/fdv | fee growth 7d | price 24h | price 7d | price 30d | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| JUP | Jupiter Perpetual Exchange | fee_growth_price_lag_candidate | 96.7959 | 0.5899 | 0.2854 | 232.55 | 3.69 | -18.29 | -35.17 | paper-label JUP as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| AAVE | Aave V3 | fee_growth_price_lag_candidate | 87.9820 | 0.9765 | 0.9265 | 129.82 | 2.79 | -21.16 | -33.32 | paper-label AAVE as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| UNI | Uniswap V3 | fee_growth_price_lag_candidate | 81.1041 | 0.3365 | 0.2341 | 126.91 | 0.29 | -15.24 | -31.19 | paper-label UNI as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| ENA | Ethena USDe | fee_price_context_watch | 60.4545 | 0.4472 | 0.2771 | 27.49 | -2.34 | 1.23 | -32.17 | collect another ENA fee and price snapshot before promotion |
| CRV | Curve DEX | fee_growth_price_lag_candidate | 57.6620 | 0.1826 | 0.1164 | 222.41 | 5.72 | -5.51 | -19.60 | paper-label CRV as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| AERO | Aerodrome Slipstream | fee_price_context_watch | 54.4518 | 0.3446 | 0.1708 | 73.72 | 3.52 | -16.20 | -29.79 | collect another AERO fee and price snapshot before promotion |
| MORPHO | Morpho Blue | fee_growth_price_lag_candidate | 51.8802 | 0.1742 | 0.1125 | 140.49 | 5.86 | -9.16 | -14.19 | paper-label MORPHO as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| PENDLE | Pendle | fee_growth_price_lag_candidate | 39.4083 | 0.1160 | 0.0703 | 158.66 | 4.18 | -4.91 | -37.48 | paper-label PENDLE as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| UNI | Uniswap V4 | fee_decay_price_weakness_context | 32.7397 | 0.1533 | 0.1067 | -32.64 | 0.29 | -15.24 | -31.19 | test whether UNI fee decay and weak price persist before any short thesis |
| HYPE | Hyperliquid Perps | fee_price_context_watch | 20.0699 | 0.0724 | 0.0169 | 111.41 | 6.57 | -16.02 | 40.94 | collect another HYPE fee and price snapshot before promotion |
| SOL | Solana | fee_decay_price_weakness_context | 10.7340 | 0.0082 | 0.0075 | -8.33 | 3.65 | -18.14 | -29.13 | test whether SOL fee decay and weak price persist before any short thesis |
| ETH | Ethereum | fee_price_context_watch | 8.1220 | 0.0016 | 0.0016 | 78.11 | 4.25 | -15.23 | -27.49 | collect another ETH fee and price snapshot before promotion |
| HYPE | Hyper Foundation HYPE Staking | fee_price_context_watch | 2.5571 | 0.0190 | 0.0044 | 2.11 | 6.57 | -16.02 | 40.94 | collect another HYPE fee and price snapshot before promotion |

## Interpretation

`fee_growth_price_lag_candidate` is the most interesting long setup class here: fees are strong, but the token has not obviously chased over the last week. `fee_growth_price_chase_risk` may still work, but it needs stricter entry timing and drawdown control.
