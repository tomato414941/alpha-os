# Current Deribit Options Realized Vol Labels

This joins Deribit ATM IV to recent Hyperliquid 15m realized volatility. It is a fast IV-vs-realized context label, not an options backtest.

| currency | expiry | dte | action | atm iv | rv 4h | rv 24h | prem 4h | prem 24h | skew | term | score |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETH | 2026-06-09 | 0.92 | cheap_vol_watch | 72.63 | 43.92 | 109.64 | 28.71 | -37.01 | 31.05 | -1.31 | 69.3662 |
| ETH | 2026-06-10 | 1.92 | cheap_vol_watch | 73.94 | 43.92 | 109.64 | 30.02 | -35.70 | 24.45 | -0.63 | 60.7762 |
| ETH | 2026-06-12 | 3.92 | cheap_vol_watch | 74.19 | 43.92 | 109.64 | 30.27 | -35.45 | 16.26 | 5.95 | 57.6562 |
| ETH | 2026-06-26 | 17.92 | cheap_vol_watch | 64.41 | 43.92 | 109.64 | 20.49 | -45.23 | 5.65 | 5.18 | 56.0562 |
| ETH | 2026-09-25 | 108.92 | cheap_vol_watch | 58.01 | 43.92 | 109.64 | 14.09 | -51.63 | 1.69 | -1.65 | 54.9662 |
| ETH | 2026-06-11 | 2.92 | cheap_vol_watch | 74.57 | 43.92 | 109.64 | 30.65 | -35.07 | 18.58 | 0.38 | 54.0262 |
| ETH | 2026-07-31 | 52.92 | cheap_vol_watch | 59.23 | 43.92 | 109.64 | 15.31 | -50.41 | 2.64 | 0.71 | 53.7562 |
| ETH | 2026-08-28 | 80.92 | cheap_vol_watch | 58.52 | 43.92 | 109.64 | 14.60 | -51.12 | 1.74 | 0.51 | 53.3662 |
| ETH | 2026-06-19 | 10.92 | cheap_vol_watch | 68.24 | 43.92 | 109.64 | 24.32 | -41.40 | 7.40 | 3.83 | 52.6262 |
| ETH | 2026-12-25 | 199.92 | cheap_vol_watch | 59.66 | 43.92 | 109.64 | 15.74 | -49.98 | 1.28 | -1.23 | 52.4862 |
| BTC | 2026-06-09 | 0.92 | cheap_vol_watch | 58.68 | 30.40 | 81.93 | 28.28 | -23.25 | 27.54 | 1.07 | 51.8621 |
| BTC | 2026-06-12 | 3.92 | cheap_vol_watch | 55.10 | 30.40 | 81.93 | 24.70 | -26.83 | 18.82 | 5.67 | 51.3221 |
| ETH | 2027-03-26 | 290.92 | cheap_vol_watch | 60.89 | 43.92 | 109.64 | 16.97 | -48.75 | 0.84 |  | 49.5862 |
| BTC | 2026-06-10 | 1.92 | cheap_vol_watch | 57.61 | 30.40 | 81.93 | 27.21 | -24.32 | 24.24 | 0.81 | 49.3721 |
| BTC | 2026-06-11 | 2.92 | cheap_vol_watch | 56.80 | 30.40 | 81.93 | 26.40 | -25.13 | 20.14 | 1.70 | 46.9721 |
| BTC | 2026-06-26 | 17.92 | cheap_vol_watch | 46.61 | 30.40 | 81.93 | 16.21 | -35.32 | 8.17 | 3.31 | 46.8021 |
| BTC | 2026-06-19 | 10.92 | cheap_vol_watch | 49.43 | 30.40 | 81.93 | 19.03 | -32.50 | 10.50 | 2.82 | 45.8221 |
| BTC | 2026-07-31 | 52.92 | cheap_vol_watch | 43.30 | 30.40 | 81.93 | 12.90 | -38.63 | 4.26 | 0.23 | 43.1221 |
| BTC | 2026-09-25 | 108.92 | cheap_vol_watch | 43.16 | 30.40 | 81.93 | 12.76 | -38.77 | 2.50 | -1.31 | 42.5821 |
| BTC | 2026-08-28 | 80.92 | cheap_vol_watch | 43.07 | 30.40 | 81.93 | 12.67 | -38.86 | 2.36 | -0.09 | 41.3121 |

## Interpretation

Positive IV premium means listed ATM IV is above recent realized volatility. This can point to vol-selling or event-premium candidates, but it still needs realized-vol forecasts, hedge PnL, option spreads, margin, and tail-risk controls.
