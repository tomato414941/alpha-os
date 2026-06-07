# Current Deribit Options Realized Vol Labels

This joins Deribit ATM IV to recent Hyperliquid 15m realized volatility. It is a fast IV-vs-realized context label, not an options backtest.

| currency | expiry | dte | action | atm iv | rv 4h | rv 24h | prem 4h | prem 24h | skew | term | score |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 2026-06-08 | 0.49 | rich_put_skew_vol_premium_watch | 75.87 | 50.32 | 47.87 | 25.55 | 28.00 | 32.77 | -7.43 | 68.1963 |
| BTC | 2026-06-09 | 1.49 | rich_put_skew_vol_premium_watch | 83.30 | 50.32 | 47.87 | 32.98 | 35.43 | 23.22 | 5.49 | 64.1363 |
| BTC | 2026-06-10 | 2.49 | rich_put_skew_vol_premium_watch | 77.81 | 50.32 | 47.87 | 27.49 | 29.94 | 21.50 | 5.01 | 56.4463 |
| ETH | 2026-06-08 | 0.49 | rich_put_skew_vol_premium_watch | 89.46 | 59.29 | 65.07 | 30.17 | 24.39 | 27.41 | -0.43 | 52.2299 |
| BTC | 2026-06-12 | 4.49 | rich_put_skew_vol_premium_watch | 68.36 | 50.32 | 47.87 | 18.04 | 20.49 | 18.74 | 12.42 | 51.6463 |
| BTC | 2026-06-11 | 3.49 | rich_put_skew_vol_premium_watch | 72.80 | 50.32 | 47.87 | 22.48 | 24.93 | 18.38 | 4.44 | 47.7463 |
| ETH | 2026-06-09 | 1.49 | rich_put_skew_vol_premium_watch | 89.89 | 59.29 | 65.07 | 30.60 | 24.82 | 14.05 | 4.07 | 42.9399 |
| ETH | 2026-06-10 | 2.49 | rich_put_skew_vol_premium_watch | 85.82 | 59.29 | 65.07 | 26.53 | 20.75 | 11.62 | 2.57 | 34.9399 |
| ETH | 2026-06-12 | 4.49 | term_structure_watch | 81.25 | 59.29 | 65.07 | 21.96 | 16.18 | 9.65 | 8.45 | 34.2799 |
| ETH | 2026-06-11 | 3.49 | realized_vol_context | 83.25 | 59.29 | 65.07 | 23.96 | 18.18 | 9.61 | 2.00 | 29.7899 |
| BTC | 2026-06-19 | 11.49 | realized_vol_context | 55.94 | 50.32 | 47.87 | 5.62 | 8.07 | 10.76 | 4.71 | 23.5363 |
| ETH | 2026-06-19 | 11.49 | realized_vol_context | 72.80 | 59.29 | 65.07 | 13.51 | 7.73 | 6.42 | 4.50 | 18.6499 |
| BTC | 2026-06-26 | 18.49 | term_structure_watch | 51.23 | 50.32 | 47.87 | 0.91 | 3.36 | 6.43 | 5.12 | 14.9063 |
| ETH | 2026-06-26 | 18.49 | term_structure_watch | 68.30 | 59.29 | 65.07 | 9.01 | 3.23 | 4.55 | 6.74 | 14.5199 |
| ETH | 2026-09-25 | 109.49 | realized_vol_context | 59.93 | 59.29 | 65.07 | 0.64 | -5.14 | 1.96 | -0.86 | 7.9601 |
| ETH | 2026-07-31 | 53.49 | realized_vol_context | 61.56 | 59.29 | 65.07 | 2.27 | -3.51 | 2.17 | 1.83 | 7.5101 |
| ETH | 2026-08-28 | 81.49 | realized_vol_context | 59.73 | 59.29 | 65.07 | 0.44 | -5.34 | 1.65 | -0.20 | 7.1901 |
| BTC | 2026-07-31 | 53.49 | realized_vol_context | 46.11 | 50.32 | 47.87 | -4.21 | -1.76 | 3.37 | 1.16 | 6.2937 |
| BTC | 2026-09-25 | 109.49 | realized_vol_context | 44.76 | 50.32 | 47.87 | -5.56 | -3.11 | 1.63 | -0.89 | 5.6337 |
| BTC | 2026-08-28 | 81.49 | realized_vol_context | 44.95 | 50.32 | 47.87 | -5.37 | -2.92 | 2.52 | 0.19 | 5.6337 |

## Interpretation

Positive IV premium means listed ATM IV is above recent realized volatility. This can point to vol-selling or event-premium candidates, but it still needs realized-vol forecasts, hedge PnL, option spreads, margin, and tail-risk controls.
