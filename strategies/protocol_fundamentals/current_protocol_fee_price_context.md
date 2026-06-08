# Current Protocol Fee Price Context

This joins protocol fee-growth valuation to current CoinGecko price movement. It looks for fee growth that is not yet fully chased by price, and for crowded price-confirmed setups.

| token | protocol | status | score | fee/mcap | fee/fdv | fee growth 7d | price 24h | price 7d | price 30d | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| JUP | Jupiter Perpetual Exchange | fee_growth_price_lag_candidate | 94.6419 | 0.5797 | 0.2805 | 232.55 | 3.79 | -13.34 | -36.40 | paper-label JUP as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| AAVE | Aave V3 | fee_growth_price_lag_candidate | 87.9820 | 0.9634 | 0.9140 | 129.82 | 3.00 | -18.98 | -31.59 | paper-label AAVE as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| UNI | Uniswap V3 | fee_growth_price_lag_candidate | 77.7421 | 0.3293 | 0.2292 | 126.91 | 1.96 | -12.13 | -29.57 | paper-label UNI as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| ENA | Ethena USDe | fee_price_context_watch | 60.1558 | 0.4423 | 0.2741 | 27.49 | -3.12 | 3.52 | -32.07 | collect another ENA fee and price snapshot before promotion |
| CRV | Curve DEX | fee_growth_price_lag_candidate | 56.5404 | 0.1826 | 0.1164 | 222.41 | 4.33 | -4.39 | -18.51 | paper-label CRV as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| AERO | Aerodrome Slipstream | fee_price_context_watch | 54.1770 | 0.3391 | 0.1681 | 73.72 | 4.66 | -11.86 | -30.00 | collect another AERO fee and price snapshot before promotion |
| MORPHO | Morpho Blue | fee_growth_price_lag_candidate | 44.6764 | 0.1633 | 0.1055 | 140.49 | 12.43 | -3.75 | -9.46 | paper-label MORPHO as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| PENDLE | Pendle | fee_growth_price_lag_candidate | 37.6100 | 0.1147 | 0.0695 | 158.66 | 3.45 | -3.32 | -35.66 | paper-label PENDLE as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| UNI | Uniswap V4 | fee_decay_price_weakness_context | 32.1877 | 0.1501 | 0.1044 | -32.64 | 1.96 | -12.13 | -29.57 | test whether UNI fee decay and weak price persist before any short thesis |
| HYPE | Hyperliquid Perps | fee_price_context_watch | 19.6759 | 0.0692 | 0.0161 | 111.41 | 10.38 | -10.43 | 48.30 | collect another HYPE fee and price snapshot before promotion |
| SOL | Solana | fee_decay_price_weakness_context | 10.7096 | 0.0080 | 0.0074 | -8.33 | 4.04 | -15.60 | -27.84 | test whether SOL fee decay and weak price persist before any short thesis |
| ETH | Ethereum | fee_price_context_watch | 8.1186 | 0.0015 | 0.0015 | 78.11 | 4.45 | -13.63 | -26.64 | collect another ETH fee and price snapshot before promotion |
| HYPE | Hyper Foundation HYPE Staking | fee_price_context_watch | 2.4536 | 0.0182 | 0.0042 | 2.11 | 10.38 | -10.43 | 48.30 | collect another HYPE fee and price snapshot before promotion |

## Interpretation

`fee_growth_price_lag_candidate` is the most interesting long setup class here: fees are strong, but the token has not obviously chased over the last week. `fee_growth_price_chase_risk` may still work, but it needs stricter entry timing and drawdown control.
