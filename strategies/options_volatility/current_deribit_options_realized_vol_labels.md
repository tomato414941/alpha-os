# Current Deribit Options Realized Vol Labels

This joins Deribit ATM IV to recent Hyperliquid 15m realized volatility. It is a fast IV-vs-realized context label, not an options backtest.

| currency | expiry | dte | action | atm iv | rv 4h | rv 24h | prem 4h | prem 24h | skew | term | score |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 2026-06-09 | 0.36 | cheap_vol_watch | 38.51 | 49.22 | 56.81 | -10.71 | -18.30 | 13.24 | -7.32 | 38.8621 |
| BTC | 2026-06-10 | 1.36 | cheap_vol_watch | 45.83 | 49.22 | 56.81 | -3.39 | -10.98 | 14.36 | -1.61 | 26.9521 |
| BTC | 2026-06-12 | 3.36 | realized_vol_context | 47.45 | 49.22 | 56.81 | -1.77 | -9.36 | 14.22 | 2.68 | 26.2621 |
| ETH | 2026-06-10 | 1.36 | realized_vol_context | 61.42 | 85.09 | 71.03 | -23.67 | -9.61 | 12.89 | -1.43 | 23.9279 |
| BTC | 2026-06-11 | 2.36 | realized_vol_context | 47.44 | 49.22 | 56.81 | -1.78 | -9.37 | 13.61 | -0.01 | 22.9921 |
| ETH | 2026-06-12 | 3.36 | realized_vol_context | 62.67 | 85.09 | 71.03 | -22.42 | -8.36 | 10.68 | 3.01 | 22.0479 |
| BTC | 2026-06-19 | 10.36 | cheap_vol_watch | 44.77 | 49.22 | 56.81 | -4.45 | -12.04 | 8.07 | 1.64 | 21.7521 |
| BTC | 2026-06-26 | 17.36 | cheap_vol_watch | 43.13 | 49.22 | 56.81 | -6.09 | -13.68 | 6.63 | 0.71 | 21.0221 |
| ETH | 2026-06-11 | 2.36 | realized_vol_context | 62.85 | 85.09 | 71.03 | -22.24 | -8.18 | 10.98 | 0.18 | 19.3379 |
| ETH | 2026-06-26 | 17.36 | cheap_vol_watch | 58.29 | 85.09 | 71.03 | -26.80 | -12.74 | 3.85 | 2.09 | 18.6779 |
| BTC | 2026-09-25 | 108.36 | cheap_vol_watch | 42.40 | 49.22 | 56.81 | -6.82 | -14.41 | 2.40 | -1.79 | 18.6021 |
| ETH | 2026-06-09 | 0.36 | cheap_vol_watch | 59.61 | 85.09 | 71.03 | -25.48 | -11.42 | 5.09 | -1.81 | 18.3179 |
| ETH | 2026-09-25 | 108.36 | cheap_vol_watch | 56.75 | 85.09 | 71.03 | -28.34 | -14.28 | 1.65 | -2.01 | 17.9379 |
| BTC | 2026-07-31 | 52.36 | cheap_vol_watch | 42.42 | 49.22 | 56.81 | -6.80 | -14.39 | 3.00 | 0.01 | 17.4021 |
| ETH | 2026-06-19 | 10.36 | cheap_vol_watch | 59.66 | 85.09 | 71.03 | -25.43 | -11.37 | 4.43 | 1.37 | 17.1679 |
| ETH | 2026-07-31 | 52.36 | cheap_vol_watch | 56.20 | 85.09 | 71.03 | -28.89 | -14.83 | 2.05 | -0.21 | 17.0879 |
| BTC | 2026-08-28 | 80.36 | cheap_vol_watch | 42.41 | 49.22 | 56.81 | -6.81 | -14.40 | 2.16 | 0.01 | 16.5721 |
| ETH | 2026-08-28 | 80.36 | cheap_vol_watch | 56.41 | 85.09 | 71.03 | -28.68 | -14.62 | 1.50 | -0.34 | 16.4579 |
| BTC | 2026-12-25 | 199.36 | cheap_vol_watch | 44.19 | 49.22 | 56.81 | -5.03 | -12.62 | 1.16 | -0.52 | 14.3021 |
| ETH | 2026-12-25 | 199.36 | cheap_vol_watch | 58.76 | 85.09 | 71.03 | -26.33 | -12.27 | 0.55 | -0.91 | 13.7279 |

## Interpretation

Positive IV premium means listed ATM IV is above recent realized volatility. This can point to vol-selling or event-premium candidates, but it still needs realized-vol forecasts, hedge PnL, option spreads, margin, and tail-risk controls.
