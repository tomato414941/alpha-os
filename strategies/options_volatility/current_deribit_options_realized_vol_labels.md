# Current Deribit Options Realized Vol Labels

This joins Deribit ATM IV to recent Hyperliquid 15m realized volatility. It is a fast IV-vs-realized context label, not an options backtest.

| currency | expiry | dte | action | atm iv | rv 4h | rv 24h | prem 4h | prem 24h | skew | term | score |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETH | 2026-06-08 | 0.33 | cheap_vol_watch | 84.65 | 231.67 | 112.34 | -147.02 | -27.69 | 30.50 | -4.11 | 62.3020 |
| ETH | 2026-06-26 | 18.33 | cheap_vol_watch | 65.14 | 231.67 | 112.34 | -166.53 | -47.20 | 4.81 | 5.30 | 57.3120 |
| ETH | 2026-09-25 | 109.33 | cheap_vol_watch | 58.90 | 231.67 | 112.34 | -172.77 | -53.44 | 1.67 | -1.30 | 56.4120 |
| ETH | 2026-07-31 | 53.33 | cheap_vol_watch | 59.84 | 231.67 | 112.34 | -171.83 | -52.50 | 2.38 | 0.82 | 55.7020 |
| BTC | 2026-06-08 | 0.33 | cheap_vol_watch | 65.15 | 166.74 | 81.41 | -101.59 | -16.26 | 26.60 | -12.28 | 55.1356 |
| ETH | 2026-08-28 | 81.33 | cheap_vol_watch | 59.02 | 231.67 | 112.34 | -172.65 | -53.32 | 1.65 | 0.12 | 55.0920 |
| ETH | 2026-12-25 | 200.33 | cheap_vol_watch | 60.20 | 231.67 | 112.34 | -171.47 | -52.14 | 1.16 | -1.12 | 54.4220 |
| ETH | 2026-06-12 | 4.33 | cheap_vol_watch | 79.58 | 231.67 | 112.34 | -152.09 | -32.76 | 10.17 | 10.97 | 53.9020 |
| ETH | 2026-06-19 | 11.33 | cheap_vol_watch | 68.61 | 231.67 | 112.34 | -163.06 | -43.73 | 6.17 | 3.47 | 53.3720 |
| ETH | 2027-03-26 | 291.33 | cheap_vol_watch | 61.32 | 231.67 | 112.34 | -170.35 | -51.02 | 0.96 |  | 51.9820 |
| ETH | 2026-06-09 | 1.33 | cheap_vol_watch | 88.76 | 231.67 | 112.34 | -142.91 | -23.58 | 20.58 | 3.42 | 47.5820 |
| ETH | 2026-06-10 | 2.33 | cheap_vol_watch | 85.34 | 231.67 | 112.34 | -146.33 | -27.00 | 15.03 | 3.43 | 45.4620 |
| ETH | 2026-06-11 | 3.33 | cheap_vol_watch | 81.91 | 231.67 | 112.34 | -149.76 | -30.43 | 12.48 | 2.33 | 45.2370 |
| BTC | 2026-06-26 | 18.33 | cheap_vol_watch | 49.27 | 166.74 | 81.41 | -117.47 | -32.14 | 6.76 | 4.47 | 43.3656 |
| BTC | 2026-06-12 | 4.33 | cheap_vol_watch | 64.91 | 166.74 | 81.41 | -101.83 | -16.50 | 14.92 | 11.43 | 42.8456 |
| BTC | 2026-07-31 | 53.33 | cheap_vol_watch | 44.80 | 166.74 | 81.41 | -121.94 | -36.61 | 3.97 | 0.98 | 41.5556 |
| BTC | 2026-06-19 | 11.33 | cheap_vol_watch | 53.48 | 166.74 | 81.41 | -113.26 | -27.93 | 9.06 | 4.21 | 41.1956 |
| BTC | 2026-08-28 | 81.33 | cheap_vol_watch | 43.82 | 166.74 | 81.41 | -122.92 | -37.59 | 2.89 | -0.12 | 40.5956 |
| BTC | 2026-09-25 | 109.33 | cheap_vol_watch | 43.94 | 166.74 | 81.41 | -122.80 | -37.47 | 1.93 | -0.98 | 40.3756 |
| BTC | 2026-12-25 | 200.33 | cheap_vol_watch | 44.92 | 166.74 | 81.41 | -121.82 | -36.49 | 1.14 | -0.52 | 38.1456 |

## Interpretation

Positive IV premium means listed ATM IV is above recent realized volatility. This can point to vol-selling or event-premium candidates, but it still needs realized-vol forecasts, hedge PnL, option spreads, margin, and tail-risk controls.
