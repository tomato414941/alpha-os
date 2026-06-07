# Current Deribit Options Realized Vol Labels

This joins Deribit ATM IV to recent Hyperliquid 15m realized volatility. It is a fast IV-vs-realized context label, not an options backtest.

| currency | expiry | dte | action | atm iv | rv 4h | rv 24h | prem 4h | prem 24h | skew | term | score |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 2026-06-09 | 1.63 | rich_put_skew_vol_premium_watch | 73.94 | 33.89 | 44.39 | 40.05 | 29.55 | 20.32 | 4.42 | 54.2895 |
| BTC | 2026-06-08 | 0.63 | rich_put_skew_vol_premium_watch | 64.68 | 33.89 | 44.39 | 30.79 | 20.29 | 20.42 | -9.26 | 49.9695 |
| BTC | 2026-06-10 | 2.63 | rich_put_skew_vol_premium_watch | 69.52 | 33.89 | 44.39 | 35.63 | 25.13 | 19.06 | 2.96 | 47.1495 |
| ETH | 2026-06-08 | 0.63 | rich_put_skew_vol_premium_watch | 84.72 | 43.20 | 63.94 | 41.52 | 20.78 | 23.44 | -2.71 | 46.9327 |
| BTC | 2026-06-12 | 4.63 | term_structure_watch | 64.15 | 33.89 | 44.39 | 30.26 | 19.76 | 15.46 | 10.39 | 45.6095 |
| ETH | 2026-06-09 | 1.63 | rich_put_skew_vol_premium_watch | 87.43 | 43.20 | 63.94 | 44.23 | 23.49 | 14.77 | 3.72 | 41.9827 |
| BTC | 2026-06-11 | 3.63 | rich_put_skew_vol_premium_watch | 66.56 | 33.89 | 44.39 | 32.67 | 22.17 | 15.13 | 2.41 | 39.7095 |
| ETH | 2026-06-12 | 4.63 | term_structure_watch | 80.84 | 43.20 | 63.94 | 37.64 | 16.90 | 10.33 | 10.91 | 38.1427 |
| ETH | 2026-06-10 | 2.63 | realized_vol_context | 83.71 | 43.20 | 63.94 | 40.51 | 19.77 | 11.64 | 1.59 | 33.0027 |
| ETH | 2026-06-11 | 3.63 | realized_vol_context | 82.12 | 43.20 | 63.94 | 38.92 | 18.18 | 10.19 | 1.28 | 29.6527 |
| BTC | 2026-06-19 | 11.63 | realized_vol_context | 53.76 | 33.89 | 44.39 | 19.87 | 9.37 | 8.96 | 4.06 | 22.3895 |
| BTC | 2026-06-26 | 18.63 | realized_vol_context | 49.70 | 33.89 | 44.39 | 15.81 | 5.31 | 6.69 | 4.50 | 16.4995 |
| ETH | 2026-06-19 | 11.63 | realized_vol_context | 69.93 | 43.20 | 63.94 | 26.73 | 5.99 | 6.37 | 3.85 | 16.2127 |
| ETH | 2026-06-26 | 18.63 | term_structure_watch | 66.08 | 43.20 | 63.94 | 22.88 | 2.14 | 4.72 | 5.60 | 12.4627 |
| ETH | 2026-07-31 | 53.63 | realized_vol_context | 60.48 | 43.20 | 63.94 | 17.28 | -3.46 | 2.33 | 1.03 | 6.8173 |
| ETH | 2026-08-28 | 81.63 | realized_vol_context | 59.45 | 43.20 | 63.94 | 16.25 | -4.49 | 1.70 | -0.06 | 6.2473 |
| ETH | 2026-12-25 | 200.63 | realized_vol_context | 59.92 | 43.20 | 63.94 | 16.72 | -4.02 | 0.58 | -1.30 | 5.8973 |
| BTC | 2026-07-31 | 53.63 | realized_vol_context | 45.20 | 33.89 | 44.39 | 11.31 | 0.81 | 3.74 | 1.25 | 5.7995 |
| ETH | 2026-09-25 | 109.63 | realized_vol_context | 59.51 | 43.20 | 63.94 | 16.31 | -4.43 | 0.71 | -0.41 | 5.5473 |
| BTC | 2026-08-28 | 81.63 | realized_vol_context | 43.95 | 33.89 | 44.39 | 10.06 | -0.44 | 2.87 | -0.44 | 3.7505 |

## Interpretation

Positive IV premium means listed ATM IV is above recent realized volatility. This can point to vol-selling or event-premium candidates, but it still needs realized-vol forecasts, hedge PnL, option spreads, margin, and tail-risk controls.
