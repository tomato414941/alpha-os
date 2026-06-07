# Current Deribit Options Realized Vol Labels

This joins Deribit ATM IV to recent Hyperliquid 15m realized volatility. It is a fast IV-vs-realized context label, not an options backtest.

| currency | expiry | dte | action | atm iv | rv 4h | rv 24h | prem 4h | prem 24h | skew | term | score |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 2026-06-08 | 0.41 | term_structure_watch | 67.77 | 51.75 | 48.08 | 16.02 | 19.69 | 31.73 | -12.30 | 63.7183 |
| BTC | 2026-06-09 | 1.41 | rich_put_skew_vol_premium_watch | 80.07 | 51.75 | 48.08 | 28.32 | 31.99 | 23.81 | 4.36 | 60.1583 |
| ETH | 2026-06-08 | 0.41 | rich_put_skew_vol_premium_watch | 87.79 | 64.08 | 65.34 | 23.71 | 22.45 | 30.62 | -2.28 | 55.3499 |
| BTC | 2026-06-10 | 2.41 | rich_put_skew_vol_premium_watch | 75.71 | 51.75 | 48.08 | 23.96 | 27.63 | 21.59 | 4.54 | 53.7583 |
| BTC | 2026-06-12 | 4.41 | term_structure_watch | 67.70 | 51.75 | 48.08 | 15.95 | 19.62 | 14.77 | 12.30 | 46.6883 |
| ETH | 2026-06-09 | 1.41 | rich_put_skew_vol_premium_watch | 90.07 | 64.08 | 65.34 | 25.99 | 24.73 | 17.36 | 3.49 | 45.5799 |
| BTC | 2026-06-11 | 3.41 | rich_put_skew_vol_premium_watch | 71.17 | 51.75 | 48.08 | 19.42 | 23.09 | 14.78 | 3.47 | 41.3383 |
| ETH | 2026-06-12 | 4.41 | term_structure_watch | 81.70 | 64.08 | 65.34 | 17.62 | 16.36 | 10.94 | 11.15 | 38.4499 |
| ETH | 2026-06-10 | 2.41 | rich_put_skew_vol_premium_watch | 86.58 | 64.08 | 65.34 | 22.50 | 21.24 | 13.26 | 2.87 | 37.3699 |
| ETH | 2026-06-11 | 3.41 | realized_vol_context | 83.71 | 64.08 | 65.34 | 19.63 | 18.37 | 11.12 | 2.01 | 31.4999 |
| BTC | 2026-06-19 | 11.41 | realized_vol_context | 55.40 | 51.75 | 48.08 | 3.65 | 7.32 | 8.60 | 4.73 | 20.6483 |
| ETH | 2026-06-19 | 11.41 | realized_vol_context | 70.55 | 64.08 | 65.34 | 6.47 | 5.21 | 6.42 | 3.82 | 15.4499 |
| BTC | 2026-06-26 | 18.41 | realized_vol_context | 50.67 | 51.75 | 48.08 | -1.08 | 2.59 | 6.50 | 4.89 | 13.9783 |
| ETH | 2026-06-26 | 18.41 | term_structure_watch | 66.73 | 64.08 | 65.34 | 2.65 | 1.39 | 5.04 | 6.15 | 12.5799 |
| ETH | 2026-07-31 | 53.41 | realized_vol_context | 60.58 | 64.08 | 65.34 | -3.50 | -4.76 | 2.43 | 1.02 | 8.2101 |
| ETH | 2026-08-28 | 81.41 | realized_vol_context | 59.56 | 64.08 | 65.34 | -4.52 | -5.78 | 1.61 | -0.25 | 7.6401 |
| ETH | 2026-12-25 | 200.41 | realized_vol_context | 60.19 | 64.08 | 65.34 | -3.89 | -5.15 | 0.55 | -1.07 | 6.7701 |
| BTC | 2026-07-31 | 53.41 | realized_vol_context | 45.78 | 51.75 | 48.08 | -5.97 | -2.30 | 3.46 | 0.95 | 6.7117 |
| ETH | 2026-09-25 | 109.41 | realized_vol_context | 59.81 | 64.08 | 65.34 | -4.27 | -5.53 | 0.80 | -0.38 | 6.7101 |
| BTC | 2026-08-28 | 81.41 | realized_vol_context | 44.83 | 51.75 | 48.08 | -6.92 | -3.25 | 2.56 | 0.14 | 5.9517 |

## Interpretation

Positive IV premium means listed ATM IV is above recent realized volatility. This can point to vol-selling or event-premium candidates, but it still needs realized-vol forecasts, hedge PnL, option spreads, margin, and tail-risk controls.
