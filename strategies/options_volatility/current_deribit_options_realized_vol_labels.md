# Current Deribit Options Realized Vol Labels

This joins Deribit ATM IV to recent Hyperliquid 15m realized volatility. It is a fast IV-vs-realized context label, not an options backtest.

| currency | expiry | dte | action | atm iv | rv 4h | rv 24h | prem 4h | prem 24h | skew | term | score |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETH | 2026-06-26 | 18.30 | cheap_vol_watch | 65.17 | 231.92 | 113.14 | -166.75 | -47.97 | 5.70 | 5.48 | 59.1538 |
| ETH | 2026-09-25 | 109.30 | cheap_vol_watch | 58.65 | 231.92 | 113.14 | -173.27 | -54.49 | 1.65 | -1.30 | 57.4438 |
| ETH | 2026-07-31 | 53.30 | cheap_vol_watch | 59.69 | 231.92 | 113.14 | -172.23 | -53.45 | 2.91 | 0.79 | 57.1538 |
| ETH | 2026-06-08 | 0.30 | cheap_vol_watch | 81.27 | 231.92 | 113.14 | -150.65 | -31.87 | 18.75 | -5.68 | 56.3038 |
| ETH | 2026-08-28 | 81.30 | cheap_vol_watch | 58.90 | 231.92 | 113.14 | -173.02 | -54.24 | 1.41 | 0.25 | 55.9038 |
| ETH | 2026-12-25 | 200.30 | cheap_vol_watch | 59.95 | 231.92 | 113.14 | -171.97 | -53.19 | 1.14 | -0.78 | 55.1138 |
| ETH | 2026-06-12 | 4.30 | cheap_vol_watch | 77.83 | 231.92 | 113.14 | -154.09 | -35.31 | 9.87 | 9.34 | 54.5238 |
| ETH | 2026-06-19 | 11.30 | cheap_vol_watch | 68.49 | 231.92 | 113.14 | -163.43 | -44.65 | 6.52 | 3.32 | 54.4938 |
| ETH | 2027-03-26 | 291.30 | cheap_vol_watch | 60.73 | 231.92 | 113.14 | -171.19 | -52.41 | 0.36 |  | 52.7738 |
| ETH | 2026-06-11 | 3.30 | cheap_vol_watch | 80.73 | 231.92 | 113.14 | -151.19 | -32.41 | 11.69 | 2.90 | 47.0038 |
| ETH | 2026-06-09 | 1.30 | cheap_vol_watch | 86.95 | 231.92 | 113.14 | -144.97 | -26.19 | 17.62 | 3.15 | 46.9638 |
| ETH | 2026-06-10 | 2.30 | cheap_vol_watch | 83.80 | 231.92 | 113.14 | -148.12 | -29.34 | 14.16 | 3.07 | 46.5738 |
| BTC | 2026-06-26 | 18.30 | cheap_vol_watch | 47.74 | 166.94 | 81.84 | -119.20 | -34.10 | 6.11 | 3.69 | 43.8978 |
| BTC | 2026-06-12 | 4.30 | cheap_vol_watch | 61.14 | 166.94 | 81.84 | -105.80 | -20.70 | 12.68 | 9.75 | 43.1278 |
| BTC | 2026-06-19 | 11.30 | cheap_vol_watch | 51.39 | 166.94 | 81.84 | -115.55 | -30.45 | 7.92 | 3.65 | 42.0178 |
| BTC | 2026-06-08 | 0.30 | cheap_vol_watch | 64.34 | 166.94 | 81.84 | -102.60 | -17.50 | 14.96 | -9.12 | 41.5778 |
| BTC | 2026-07-31 | 53.30 | cheap_vol_watch | 44.05 | 166.94 | 81.84 | -122.89 | -37.79 | 3.30 | 0.38 | 41.4678 |
| BTC | 2026-08-28 | 81.30 | cheap_vol_watch | 43.67 | 166.94 | 81.84 | -123.27 | -38.17 | 2.30 | -0.13 | 40.5978 |
| BTC | 2026-09-25 | 109.30 | cheap_vol_watch | 43.80 | 166.94 | 81.84 | -123.14 | -38.04 | 1.68 | -0.84 | 40.5578 |
| BTC | 2026-12-25 | 200.30 | cheap_vol_watch | 44.64 | 166.94 | 81.84 | -122.30 | -37.20 | 1.13 | -0.59 | 38.9178 |

## Interpretation

Positive IV premium means listed ATM IV is above recent realized volatility. This can point to vol-selling or event-premium candidates, but it still needs realized-vol forecasts, hedge PnL, option spreads, margin, and tail-risk controls.
