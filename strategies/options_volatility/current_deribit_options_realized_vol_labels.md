# Current Deribit Options Realized Vol Labels

This joins Deribit ATM IV to recent Hyperliquid 15m realized volatility. It is a fast IV-vs-realized context label, not an options backtest.

| currency | expiry | dte | action | atm iv | rv 4h | rv 24h | prem 4h | prem 24h | skew | term | score |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETH | 2026-06-09 | 0.50 | cheap_vol_watch | 58.37 | 54.16 | 112.32 | 4.21 | -53.95 | 13.33 | -4.16 | 71.4350 |
| BTC | 2026-06-09 | 0.50 | cheap_vol_watch | 38.88 | 41.33 | 85.77 | -2.45 | -46.89 | 17.14 | -6.68 | 70.7127 |
| ETH | 2026-06-10 | 1.50 | cheap_vol_watch | 62.53 | 54.16 | 112.32 | 8.37 | -49.79 | 15.46 | -2.19 | 67.4350 |
| ETH | 2026-06-12 | 3.50 | cheap_vol_watch | 64.28 | 54.16 | 112.32 | 10.12 | -48.04 | 11.42 | 4.22 | 63.6750 |
| ETH | 2026-06-26 | 17.50 | cheap_vol_watch | 58.31 | 54.16 | 112.32 | 4.15 | -54.01 | 5.09 | 2.13 | 61.2250 |
| ETH | 2026-06-11 | 2.50 | cheap_vol_watch | 64.72 | 54.16 | 112.32 | 10.56 | -47.60 | 12.97 | 0.44 | 61.0050 |
| BTC | 2026-06-10 | 1.50 | cheap_vol_watch | 45.56 | 41.33 | 85.77 | 4.23 | -40.21 | 18.18 | -2.01 | 60.4027 |
| ETH | 2026-06-19 | 10.50 | cheap_vol_watch | 60.06 | 54.16 | 112.32 | 5.90 | -52.26 | 6.15 | 1.75 | 60.1550 |
| ETH | 2026-09-25 | 108.50 | cheap_vol_watch | 56.68 | 54.16 | 112.32 | 2.52 | -55.64 | 1.56 | -2.12 | 59.3150 |
| ETH | 2026-07-31 | 52.50 | cheap_vol_watch | 56.18 | 54.16 | 112.32 | 2.02 | -56.14 | 2.48 | -0.24 | 58.8550 |
| ETH | 2026-08-28 | 80.50 | cheap_vol_watch | 56.42 | 54.16 | 112.32 | 2.26 | -55.90 | 2.05 | -0.26 | 58.2050 |
| ETH | 2026-12-25 | 199.50 | cheap_vol_watch | 58.80 | 54.16 | 112.32 | 4.64 | -53.52 | 1.15 | -1.39 | 56.0550 |
| BTC | 2026-06-12 | 3.50 | cheap_vol_watch | 47.97 | 41.33 | 85.77 | 6.64 | -37.80 | 15.86 | 1.86 | 55.5227 |
| BTC | 2026-06-11 | 2.50 | cheap_vol_watch | 47.57 | 41.33 | 85.77 | 6.24 | -38.20 | 15.90 | -0.40 | 54.5027 |
| ETH | 2027-03-26 | 290.50 | cheap_vol_watch | 60.19 | 54.16 | 112.32 | 6.03 | -52.13 | 0.94 |  | 53.0650 |
| BTC | 2026-06-26 | 17.50 | cheap_vol_watch | 44.02 | 41.33 | 85.77 | 2.69 | -41.75 | 6.75 | 2.02 | 50.5227 |
| BTC | 2026-06-19 | 10.50 | cheap_vol_watch | 46.11 | 41.33 | 85.77 | 4.78 | -39.66 | 8.65 | 2.09 | 50.4027 |
| BTC | 2026-09-25 | 108.50 | cheap_vol_watch | 42.27 | 41.33 | 85.77 | 0.94 | -43.50 | 2.48 | -1.81 | 47.7927 |
| BTC | 2026-07-31 | 52.50 | cheap_vol_watch | 42.00 | 41.33 | 85.77 | 0.67 | -43.77 | 3.78 | -0.02 | 47.5727 |
| BTC | 2026-08-28 | 80.50 | cheap_vol_watch | 42.02 | 41.33 | 85.77 | 0.69 | -43.75 | 2.39 | -0.25 | 46.3927 |

## Interpretation

Positive IV premium means listed ATM IV is above recent realized volatility. This can point to vol-selling or event-premium candidates, but it still needs realized-vol forecasts, hedge PnL, option spreads, margin, and tail-risk controls.
