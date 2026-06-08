# Current Deribit Options Realized Vol Labels

This joins Deribit ATM IV to recent Hyperliquid 15m realized volatility. It is a fast IV-vs-realized context label, not an options backtest.

| currency | expiry | dte | action | atm iv | rv 4h | rv 24h | prem 4h | prem 24h | skew | term | score |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETH | 2026-06-09 | 0.88 | cheap_vol_watch | 71.70 | 42.92 | 109.72 | 28.78 | -38.02 | 24.71 | -0.20 | 62.9288 |
| ETH | 2026-06-12 | 3.88 | cheap_vol_watch | 72.20 | 42.92 | 109.72 | 29.28 | -37.52 | 16.70 | 5.15 | 59.3688 |
| ETH | 2026-06-10 | 1.88 | cheap_vol_watch | 71.90 | 42.92 | 109.72 | 28.98 | -37.82 | 21.07 | -0.44 | 59.3288 |
| ETH | 2026-06-26 | 17.88 | cheap_vol_watch | 63.74 | 42.92 | 109.72 | 20.82 | -45.98 | 5.77 | 4.93 | 56.6788 |
| ETH | 2026-06-11 | 2.88 | cheap_vol_watch | 72.34 | 42.92 | 109.72 | 29.42 | -37.38 | 18.42 | 0.14 | 55.9388 |
| ETH | 2026-09-25 | 108.88 | cheap_vol_watch | 57.80 | 42.92 | 109.72 | 14.88 | -51.92 | 1.71 | -1.61 | 55.2388 |
| ETH | 2026-07-31 | 52.88 | cheap_vol_watch | 58.81 | 42.92 | 109.72 | 15.89 | -50.91 | 2.72 | 1.14 | 54.7688 |
| ETH | 2026-08-28 | 80.88 | cheap_vol_watch | 57.67 | 42.92 | 109.72 | 14.75 | -52.05 | 1.79 | -0.13 | 53.9688 |
| ETH | 2026-06-19 | 10.88 | cheap_vol_watch | 67.05 | 42.92 | 109.72 | 24.13 | -42.67 | 7.80 | 3.31 | 53.7788 |
| ETH | 2026-12-25 | 199.88 | cheap_vol_watch | 59.41 | 42.92 | 109.72 | 16.49 | -50.31 | 1.17 | -1.35 | 52.8288 |
| BTC | 2026-06-12 | 3.88 | cheap_vol_watch | 56.08 | 36.62 | 82.19 | 19.46 | -26.11 | 16.69 | 7.24 | 50.0443 |
| ETH | 2027-03-26 | 290.88 | cheap_vol_watch | 60.76 | 42.92 | 109.72 | 17.84 | -48.96 | 0.95 |  | 49.9088 |
| BTC | 2026-06-09 | 0.88 | cheap_vol_watch | 61.11 | 36.62 | 82.19 | 24.49 | -21.08 | 23.05 | 2.35 | 46.4843 |
| BTC | 2026-06-26 | 17.88 | cheap_vol_watch | 46.16 | 36.62 | 82.19 | 9.54 | -36.03 | 7.32 | 2.54 | 45.8943 |
| BTC | 2026-06-19 | 10.88 | cheap_vol_watch | 48.84 | 36.62 | 82.19 | 12.22 | -33.35 | 9.13 | 2.68 | 45.1643 |
| BTC | 2026-06-10 | 1.88 | cheap_vol_watch | 58.76 | 36.62 | 82.19 | 22.14 | -23.43 | 20.23 | 1.11 | 44.7743 |
| BTC | 2026-06-11 | 2.88 | cheap_vol_watch | 57.65 | 36.62 | 82.19 | 21.03 | -24.54 | 18.18 | 1.57 | 44.2943 |
| BTC | 2026-09-25 | 108.88 | cheap_vol_watch | 42.96 | 36.62 | 82.19 | 6.34 | -39.23 | 1.98 | -1.54 | 42.7543 |
| BTC | 2026-07-31 | 52.88 | cheap_vol_watch | 43.62 | 36.62 | 82.19 | 7.00 | -38.57 | 3.81 | 0.19 | 42.5743 |
| BTC | 2026-08-28 | 80.88 | cheap_vol_watch | 43.43 | 36.62 | 82.19 | 6.81 | -38.76 | 2.93 | 0.47 | 42.1643 |

## Interpretation

Positive IV premium means listed ATM IV is above recent realized volatility. This can point to vol-selling or event-premium candidates, but it still needs realized-vol forecasts, hedge PnL, option spreads, margin, and tail-risk controls.
