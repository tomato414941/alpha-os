# Current Deribit Options Realized Vol Labels

This joins Deribit ATM IV to recent Hyperliquid 15m realized volatility. It is a fast IV-vs-realized context label, not an options backtest.

| currency | expiry | dte | action | atm iv | rv 4h | rv 24h | prem 4h | prem 24h | skew | term | score |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETH | 2026-06-08 | 0.28 | cheap_vol_watch | 87.62 | 238.75 | 111.17 | -151.13 | -23.55 | 36.82 | -1.50 | 61.8690 |
| ETH | 2026-06-26 | 18.28 | cheap_vol_watch | 64.92 | 238.75 | 111.17 | -173.83 | -46.25 | 4.77 | 5.42 | 56.4390 |
| ETH | 2026-09-25 | 109.28 | cheap_vol_watch | 58.61 | 238.75 | 111.17 | -180.14 | -52.56 | 1.60 | -1.31 | 55.4690 |
| BTC | 2026-06-08 | 0.28 | cheap_vol_watch | 68.60 | 173.89 | 81.65 | -105.29 | -13.05 | 36.21 | -6.06 | 55.3223 |
| ETH | 2026-07-31 | 53.28 | cheap_vol_watch | 59.50 | 238.75 | 111.17 | -179.25 | -51.67 | 2.30 | 0.74 | 54.7090 |
| ETH | 2026-08-28 | 81.28 | cheap_vol_watch | 58.76 | 238.75 | 111.17 | -179.99 | -52.41 | 1.63 | 0.15 | 54.1890 |
| ETH | 2026-06-12 | 4.28 | cheap_vol_watch | 79.01 | 238.75 | 111.17 | -159.74 | -32.16 | 11.01 | 10.63 | 53.7990 |
| ETH | 2026-12-25 | 200.28 | cheap_vol_watch | 59.92 | 238.75 | 111.17 | -178.83 | -51.25 | 1.11 | -1.16 | 53.5190 |
| ETH | 2026-06-19 | 11.28 | cheap_vol_watch | 68.38 | 238.75 | 111.17 | -170.37 | -42.79 | 5.89 | 3.46 | 52.1390 |
| ETH | 2027-03-26 | 291.28 | cheap_vol_watch | 61.08 | 238.75 | 111.17 | -177.67 | -50.09 | 0.84 |  | 50.9290 |
| ETH | 2026-06-09 | 1.28 | cheap_vol_watch | 89.12 | 238.75 | 111.17 | -149.63 | -22.05 | 23.12 | 3.75 | 48.9190 |
| ETH | 2026-06-10 | 2.28 | cheap_vol_watch | 85.37 | 238.75 | 111.17 | -153.38 | -25.80 | 16.22 | 3.47 | 45.4890 |
| ETH | 2026-06-11 | 3.28 | cheap_vol_watch | 81.90 | 238.75 | 111.17 | -156.85 | -29.27 | 12.94 | 2.89 | 45.0990 |
| BTC | 2026-06-12 | 4.28 | cheap_vol_watch | 62.26 | 173.89 | 81.65 | -111.63 | -19.39 | 14.19 | 10.28 | 43.8623 |
| BTC | 2026-06-26 | 18.28 | cheap_vol_watch | 48.22 | 173.89 | 81.65 | -125.67 | -33.43 | 6.18 | 3.80 | 43.4123 |
| BTC | 2026-06-19 | 11.28 | cheap_vol_watch | 51.98 | 173.89 | 81.65 | -121.91 | -29.67 | 8.18 | 3.76 | 41.6123 |
| BTC | 2026-07-31 | 53.28 | cheap_vol_watch | 44.42 | 173.89 | 81.65 | -129.47 | -37.23 | 3.41 | 0.42 | 41.0623 |
| BTC | 2026-09-25 | 109.28 | cheap_vol_watch | 43.78 | 173.89 | 81.65 | -130.11 | -37.87 | 1.85 | -1.10 | 40.8223 |
| BTC | 2026-08-28 | 81.28 | cheap_vol_watch | 44.00 | 173.89 | 81.65 | -129.89 | -37.65 | 2.71 | 0.22 | 40.5823 |
| BTC | 2026-12-25 | 200.28 | cheap_vol_watch | 44.88 | 173.89 | 81.65 | -129.01 | -36.77 | 1.11 | -0.52 | 38.4023 |

## Interpretation

Positive IV premium means listed ATM IV is above recent realized volatility. This can point to vol-selling or event-premium candidates, but it still needs realized-vol forecasts, hedge PnL, option spreads, margin, and tail-risk controls.
