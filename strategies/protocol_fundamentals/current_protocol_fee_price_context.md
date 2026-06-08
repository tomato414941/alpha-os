# Current Protocol Fee Price Context

This joins protocol fee-growth valuation to current CoinGecko price movement. It looks for fee growth that is not yet fully chased by price, and for crowded price-confirmed setups.

| token | protocol | status | score | fee/mcap | fee/fdv | fee growth 7d | price 24h | price 7d | price 30d | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| JUP | Jupiter Perpetual Exchange | fee_growth_price_lag_candidate | 93.9263 | 0.5887 | 0.2848 | 204.43 | 3.55 | -16.78 | -35.70 | paper-label JUP as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| AAVE | Aave V3 | fee_growth_price_lag_candidate | 87.9800 | 0.9721 | 0.9222 | 129.80 | 5.29 | -21.82 | -32.87 | paper-label AAVE as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| UNI | Uniswap V3 | fee_growth_price_lag_candidate | 77.8233 | 0.3267 | 0.2273 | 105.16 | 4.46 | -14.58 | -30.21 | paper-label UNI as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| CRV | Curve DEX | fee_growth_price_lag_candidate | 59.9474 | 0.1855 | 0.1182 | 210.10 | 6.12 | -8.57 | -23.50 | paper-label CRV as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| ENA | Ethena USDe | fee_price_context_watch | 59.4446 | 0.4309 | 0.2670 | 27.49 | 0.71 | 2.49 | -31.83 | collect another ENA fee and price snapshot before promotion |
| MORPHO | Morpho Blue | fee_growth_price_lag_candidate | 58.5616 | 0.1792 | 0.1158 | 140.56 | 6.04 | -19.06 | -16.35 | paper-label MORPHO as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| AERO | Aerodrome Slipstream | fee_price_context_watch | 50.8194 | 0.3387 | 0.1679 | 40.32 | 8.20 | -16.56 | -24.13 | collect another AERO fee and price snapshot before promotion |
| PENDLE | Pendle | fee_growth_price_lag_candidate | 40.2142 | 0.1156 | 0.0701 | 158.55 | 5.23 | -5.79 | -37.96 | paper-label PENDLE as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| UNI | Uniswap V4 | fee_decay_price_weakness_context | 31.9772 | 0.1491 | 0.1037 | -33.04 | 4.46 | -14.58 | -30.21 | test whether UNI fee decay and weak price persist before any short thesis |
| HYPE | Hyperliquid Perps | fee_price_context_watch | 18.2341 | 0.0722 | 0.0168 | 93.34 | 7.88 | -14.98 | 42.41 | collect another HYPE fee and price snapshot before promotion |
| HYPE | Hyper Foundation HYPE Staking | fee_decay_price_weakness_context | 11.2882 | 0.0190 | 0.0044 | -10.50 | 7.88 | -14.98 | 42.41 | test whether HYPE fee decay and weak price persist before any short thesis |
| SOL | Solana | fee_decay_price_weakness_context | 9.6990 | 0.0081 | 0.0074 | -18.52 | 7.34 | -18.76 | -27.31 | test whether SOL fee decay and weak price persist before any short thesis |
| ETH | Ethereum | fee_price_context_watch | 7.6685 | 0.0015 | 0.0015 | 73.63 | 8.74 | -15.06 | -26.20 | collect another ETH fee and price snapshot before promotion |

## Interpretation

`fee_growth_price_lag_candidate` is the most interesting long setup class here: fees are strong, but the token has not obviously chased over the last week. `fee_growth_price_chase_risk` may still work, but it needs stricter entry timing and drawdown control.
