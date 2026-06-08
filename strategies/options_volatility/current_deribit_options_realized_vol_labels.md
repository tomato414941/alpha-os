# Current Deribit Options Realized Vol Labels

This joins Deribit ATM IV to recent Hyperliquid 15m realized volatility. It is a fast IV-vs-realized context label, not an options backtest.

| currency | expiry | dte | action | atm iv | rv 4h | rv 24h | prem 4h | prem 24h | skew | term | score |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETH | 2026-06-26 | 17.81 | cheap_vol_watch | 61.65 | 77.80 | 110.81 | -16.15 | -49.16 | 5.70 | 3.92 | 58.7817 |
| ETH | 2026-06-12 | 3.81 | cheap_vol_watch | 72.97 | 77.80 | 110.81 | -4.83 | -37.84 | 11.93 | 8.54 | 58.3117 |
| ETH | 2026-09-25 | 108.81 | cheap_vol_watch | 57.54 | 77.80 | 110.81 | -20.26 | -53.27 | 1.77 | -1.83 | 56.8717 |
| ETH | 2026-06-19 | 10.81 | cheap_vol_watch | 64.43 | 77.80 | 110.81 | -13.37 | -46.38 | 7.43 | 2.78 | 56.5917 |
| ETH | 2026-07-31 | 52.81 | cheap_vol_watch | 57.73 | 77.80 | 110.81 | -20.07 | -53.08 | 2.66 | 0.33 | 56.0717 |
| ETH | 2026-08-28 | 80.81 | cheap_vol_watch | 57.40 | 77.80 | 110.81 | -20.40 | -53.41 | 2.25 | -0.14 | 55.8017 |
| ETH | 2026-12-25 | 199.81 | cheap_vol_watch | 59.37 | 77.80 | 110.81 | -18.43 | -51.44 | 1.30 | -1.22 | 53.9617 |
| ETH | 2026-06-09 | 0.81 | cheap_vol_watch | 77.33 | 77.80 | 110.81 | -0.47 | -33.48 | 16.98 | 2.57 | 53.0317 |
| ETH | 2026-06-10 | 1.81 | cheap_vol_watch | 74.76 | 77.80 | 110.81 | -3.04 | -36.05 | 15.10 | 0.64 | 51.7917 |
| ETH | 2026-06-11 | 2.81 | cheap_vol_watch | 74.12 | 77.80 | 110.81 | -3.68 | -36.69 | 13.85 | 1.15 | 51.6917 |
| BTC | 2026-06-12 | 3.81 | cheap_vol_watch | 55.79 | 83.13 | 85.22 | -27.34 | -29.43 | 14.95 | 6.95 | 51.3337 |
| ETH | 2027-03-26 | 290.81 | cheap_vol_watch | 60.59 | 77.80 | 110.81 | -17.21 | -50.22 | 1.07 |  | 51.2917 |
| BTC | 2026-06-26 | 17.81 | cheap_vol_watch | 45.99 | 83.13 | 85.22 | -37.14 | -39.23 | 7.17 | 3.55 | 49.9537 |
| BTC | 2026-06-19 | 10.81 | cheap_vol_watch | 48.84 | 83.13 | 85.22 | -34.29 | -36.38 | 9.19 | 2.85 | 48.4237 |
| BTC | 2026-07-31 | 52.81 | cheap_vol_watch | 42.44 | 83.13 | 85.22 | -40.69 | -42.78 | 4.13 | -0.07 | 46.9837 |
| BTC | 2026-09-25 | 108.81 | cheap_vol_watch | 42.58 | 83.13 | 85.22 | -40.55 | -42.64 | 2.50 | -1.49 | 46.6337 |
| BTC | 2026-08-28 | 80.81 | cheap_vol_watch | 42.51 | 83.13 | 85.22 | -40.62 | -42.71 | 2.38 | -0.07 | 45.1637 |
| BTC | 2026-06-10 | 1.81 | cheap_vol_watch | 59.80 | 83.13 | 85.22 | -23.33 | -25.42 | 17.89 | 1.78 | 45.0937 |
| BTC | 2026-06-09 | 0.81 | cheap_vol_watch | 64.82 | 83.13 | 85.22 | -18.31 | -20.40 | 19.30 | 5.02 | 44.7237 |
| BTC | 2026-06-11 | 2.81 | cheap_vol_watch | 58.02 | 83.13 | 85.22 | -25.11 | -27.20 | 15.21 | 2.23 | 44.6437 |

## Interpretation

Positive IV premium means listed ATM IV is above recent realized volatility. This can point to vol-selling or event-premium candidates, but it still needs realized-vol forecasts, hedge PnL, option spreads, margin, and tail-risk controls.
