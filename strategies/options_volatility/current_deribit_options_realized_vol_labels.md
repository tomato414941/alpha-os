# Current Deribit Options Realized Vol Labels

This joins Deribit ATM IV to recent Hyperliquid 15m realized volatility. It is a fast IV-vs-realized context label, not an options backtest.

| currency | expiry | dte | action | atm iv | rv 4h | rv 24h | prem 4h | prem 24h | skew | term | score |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETH | 2026-06-09 | 0.70 | cheap_vol_watch | 61.17 | 79.70 | 111.18 | -18.53 | -50.01 | 19.46 | -3.49 | 72.9616 |
| ETH | 2026-06-10 | 1.70 | cheap_vol_watch | 64.66 | 79.70 | 111.18 | -15.04 | -46.52 | 17.01 | -1.61 | 65.1416 |
| ETH | 2026-06-12 | 3.70 | cheap_vol_watch | 65.60 | 79.70 | 111.18 | -14.10 | -45.58 | 11.81 | 4.11 | 61.5016 |
| ETH | 2026-06-26 | 17.70 | cheap_vol_watch | 59.29 | 79.70 | 111.18 | -20.41 | -51.89 | 5.55 | 2.53 | 59.9716 |
| ETH | 2026-06-11 | 2.70 | cheap_vol_watch | 66.27 | 79.70 | 111.18 | -13.43 | -44.91 | 13.85 | 0.67 | 59.4316 |
| ETH | 2026-06-19 | 10.70 | cheap_vol_watch | 61.49 | 79.70 | 111.18 | -18.21 | -49.69 | 7.07 | 2.20 | 58.9616 |
| ETH | 2026-07-31 | 52.70 | cheap_vol_watch | 56.76 | 79.70 | 111.18 | -22.94 | -54.42 | 3.20 | -0.11 | 57.7316 |
| ETH | 2026-09-25 | 108.70 | cheap_vol_watch | 57.10 | 79.70 | 111.18 | -22.60 | -54.08 | 1.60 | -2.00 | 57.6816 |
| BTC | 2026-06-09 | 0.70 | cheap_vol_watch | 49.04 | 87.19 | 85.89 | -38.15 | -36.85 | 19.75 | 0.27 | 56.8657 |
| ETH | 2026-08-28 | 80.70 | cheap_vol_watch | 56.87 | 79.70 | 111.18 | -22.83 | -54.31 | 2.20 | -0.23 | 56.7416 |
| BTC | 2026-06-10 | 1.70 | cheap_vol_watch | 48.77 | 87.19 | 85.89 | -38.42 | -37.12 | 17.95 | -1.23 | 56.2957 |
| ETH | 2026-12-25 | 199.70 | cheap_vol_watch | 59.10 | 79.70 | 111.18 | -20.60 | -52.08 | 1.17 | -1.36 | 54.6116 |
| BTC | 2026-06-12 | 3.70 | cheap_vol_watch | 48.46 | 87.19 | 85.89 | -38.73 | -37.43 | 13.19 | 3.71 | 54.3257 |
| BTC | 2026-06-11 | 2.70 | cheap_vol_watch | 50.00 | 87.19 | 85.89 | -37.19 | -35.89 | 14.70 | 1.54 | 52.1257 |
| ETH | 2027-03-26 | 290.70 | cheap_vol_watch | 60.46 | 79.70 | 111.18 | -19.24 | -50.72 | 0.45 |  | 51.1716 |
| BTC | 2026-06-19 | 10.70 | cheap_vol_watch | 44.75 | 87.19 | 85.89 | -42.44 | -41.14 | 7.57 | 2.32 | 51.0257 |
| BTC | 2026-06-26 | 17.70 | cheap_vol_watch | 42.43 | 87.19 | 85.89 | -44.76 | -43.46 | 6.04 | 1.11 | 50.6057 |
| BTC | 2026-07-31 | 52.70 | cheap_vol_watch | 41.32 | 87.19 | 85.89 | -45.87 | -44.57 | 3.46 | -0.28 | 48.3057 |
| BTC | 2026-09-25 | 108.70 | cheap_vol_watch | 41.64 | 87.19 | 85.89 | -45.55 | -44.25 | 1.74 | -2.08 | 48.0657 |
| BTC | 2026-08-28 | 80.70 | cheap_vol_watch | 41.60 | 87.19 | 85.89 | -45.59 | -44.29 | 2.74 | -0.04 | 47.0657 |

## Interpretation

Positive IV premium means listed ATM IV is above recent realized volatility. This can point to vol-selling or event-premium candidates, but it still needs realized-vol forecasts, hedge PnL, option spreads, margin, and tail-risk controls.
