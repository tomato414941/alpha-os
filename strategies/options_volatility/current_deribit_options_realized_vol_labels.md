# Current Deribit Options Realized Vol Labels

This joins Deribit ATM IV to recent Hyperliquid 15m realized volatility. It is a fast IV-vs-realized context label, not an options backtest.

| currency | expiry | dte | action | atm iv | rv 4h | rv 24h | prem 4h | prem 24h | skew | term | score |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETH | 2026-06-12 | 3.87 | cheap_vol_watch | 71.63 | 42.57 | 109.71 | 29.06 | -38.08 | 15.61 | 4.92 | 58.6117 |
| ETH | 2026-06-09 | 0.87 | cheap_vol_watch | 71.00 | 42.57 | 109.71 | 28.43 | -38.71 | 18.66 | -0.22 | 57.5917 |
| ETH | 2026-06-26 | 17.87 | cheap_vol_watch | 63.49 | 42.57 | 109.71 | 20.92 | -46.22 | 5.66 | 4.63 | 56.5117 |
| ETH | 2026-09-25 | 108.87 | cheap_vol_watch | 57.78 | 42.57 | 109.71 | 15.21 | -51.93 | 1.73 | -1.78 | 55.4417 |
| ETH | 2026-06-10 | 1.87 | cheap_vol_watch | 71.22 | 42.57 | 109.71 | 28.65 | -38.49 | 15.88 | -0.88 | 55.2517 |
| ETH | 2026-07-31 | 52.87 | cheap_vol_watch | 58.86 | 42.57 | 109.71 | 16.29 | -50.85 | 2.70 | 1.20 | 54.7517 |
| ETH | 2026-08-28 | 80.87 | cheap_vol_watch | 57.66 | 42.57 | 109.71 | 15.09 | -52.05 | 1.75 | -0.12 | 53.9217 |
| ETH | 2026-06-19 | 10.87 | cheap_vol_watch | 66.71 | 42.57 | 109.71 | 24.14 | -43.00 | 7.60 | 3.22 | 53.8217 |
| ETH | 2026-12-25 | 199.87 | cheap_vol_watch | 59.56 | 42.57 | 109.71 | 16.99 | -50.15 | 1.20 | -1.19 | 52.5417 |
| ETH | 2026-06-11 | 2.87 | cheap_vol_watch | 72.10 | 42.57 | 109.71 | 29.53 | -37.61 | 13.38 | 0.47 | 51.4617 |
| ETH | 2027-03-26 | 290.87 | cheap_vol_watch | 60.75 | 42.57 | 109.71 | 18.18 | -48.96 | 0.93 |  | 49.8917 |
| BTC | 2026-06-12 | 3.87 | cheap_vol_watch | 56.19 | 36.12 | 82.16 | 20.07 | -25.97 | 16.38 | 7.37 | 49.7243 |
| BTC | 2026-06-09 | 0.87 | cheap_vol_watch | 61.29 | 36.12 | 82.16 | 25.17 | -20.87 | 24.00 | 2.57 | 47.4443 |
| BTC | 2026-06-26 | 17.87 | cheap_vol_watch | 46.10 | 36.12 | 82.16 | 9.98 | -36.06 | 7.26 | 2.52 | 45.8443 |
| BTC | 2026-06-19 | 10.87 | cheap_vol_watch | 48.82 | 36.12 | 82.16 | 12.70 | -33.34 | 9.21 | 2.72 | 45.2743 |
| BTC | 2026-06-10 | 1.87 | cheap_vol_watch | 58.72 | 36.12 | 82.16 | 22.60 | -23.44 | 19.60 | 1.07 | 44.1143 |
| BTC | 2026-06-11 | 2.87 | cheap_vol_watch | 57.65 | 36.12 | 82.16 | 21.53 | -24.51 | 17.87 | 1.46 | 43.8443 |
| BTC | 2026-07-31 | 52.87 | cheap_vol_watch | 43.58 | 36.12 | 82.16 | 7.46 | -38.58 | 4.15 | 0.63 | 43.3643 |
| BTC | 2026-09-25 | 108.87 | cheap_vol_watch | 42.91 | 36.12 | 82.16 | 6.79 | -39.25 | 1.98 | -1.56 | 42.7943 |
| BTC | 2026-08-28 | 80.87 | cheap_vol_watch | 42.95 | 36.12 | 82.16 | 6.83 | -39.21 | 2.93 | 0.04 | 42.1843 |

## Interpretation

Positive IV premium means listed ATM IV is above recent realized volatility. This can point to vol-selling or event-premium candidates, but it still needs realized-vol forecasts, hedge PnL, option spreads, margin, and tail-risk controls.
