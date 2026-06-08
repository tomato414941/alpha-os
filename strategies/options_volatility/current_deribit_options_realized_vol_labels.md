# Current Deribit Options Realized Vol Labels

This joins Deribit ATM IV to recent Hyperliquid 15m realized volatility. It is a fast IV-vs-realized context label, not an options backtest.

| currency | expiry | dte | action | atm iv | rv 4h | rv 24h | prem 4h | prem 24h | skew | term | score |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETH | 2026-06-26 | 17.82 | cheap_vol_watch | 63.71 | 72.44 | 110.29 | -8.73 | -46.58 | 5.56 | 5.93 | 58.0715 |
| ETH | 2026-09-25 | 108.82 | cheap_vol_watch | 57.66 | 72.44 | 110.29 | -14.78 | -52.63 | 1.76 | -1.78 | 56.1715 |
| ETH | 2026-07-31 | 52.82 | cheap_vol_watch | 57.78 | 72.44 | 110.29 | -14.66 | -52.51 | 2.69 | 0.34 | 55.5415 |
| ETH | 2026-06-09 | 0.82 | cheap_vol_watch | 78.50 | 72.44 | 110.29 | 6.06 | -31.79 | 19.57 | 3.77 | 55.1315 |
| ETH | 2026-08-28 | 80.82 | cheap_vol_watch | 57.44 | 72.44 | 110.29 | -15.00 | -52.85 | 1.81 | -0.22 | 54.8815 |
| ETH | 2026-06-19 | 10.82 | cheap_vol_watch | 67.21 | 72.44 | 110.29 | -5.23 | -43.08 | 7.32 | 3.50 | 53.9015 |
| ETH | 2026-06-12 | 3.82 | cheap_vol_watch | 73.24 | 72.44 | 110.29 | 0.80 | -37.05 | 10.73 | 6.03 | 53.8115 |
| ETH | 2026-12-25 | 199.82 | cheap_vol_watch | 59.44 | 72.44 | 110.29 | -13.00 | -50.85 | 1.19 | -1.33 | 53.3715 |
| BTC | 2026-06-12 | 3.82 | cheap_vol_watch | 58.03 | 74.91 | 84.60 | -16.88 | -26.57 | 15.02 | 8.87 | 50.4559 |
| ETH | 2027-03-26 | 290.82 | cheap_vol_watch | 60.77 | 72.44 | 110.29 | -11.67 | -49.52 | 0.91 |  | 50.4315 |
| ETH | 2026-06-10 | 1.82 | cheap_vol_watch | 74.73 | 72.44 | 110.29 | 2.29 | -35.56 | 14.15 | 0.56 | 50.2715 |
| ETH | 2026-06-11 | 2.82 | cheap_vol_watch | 74.17 | 72.44 | 110.29 | 1.73 | -36.12 | 12.52 | 0.93 | 49.5715 |
| BTC | 2026-06-26 | 17.82 | cheap_vol_watch | 46.18 | 74.91 | 84.60 | -28.73 | -38.42 | 6.84 | 2.95 | 48.2059 |
| BTC | 2026-06-19 | 10.82 | cheap_vol_watch | 49.16 | 74.91 | 84.60 | -25.75 | -35.44 | 8.65 | 2.98 | 47.0659 |
| BTC | 2026-09-25 | 108.82 | cheap_vol_watch | 42.68 | 74.91 | 84.60 | -32.23 | -41.92 | 2.00 | -1.59 | 45.5059 |
| BTC | 2026-07-31 | 52.82 | cheap_vol_watch | 43.23 | 74.91 | 84.60 | -31.68 | -41.37 | 3.77 | 0.22 | 45.3559 |
| BTC | 2026-08-28 | 80.82 | cheap_vol_watch | 43.01 | 74.91 | 84.60 | -31.90 | -41.59 | 2.61 | 0.33 | 44.5259 |
| BTC | 2026-06-09 | 0.82 | cheap_vol_watch | 70.62 | 74.91 | 84.60 | -4.29 | -13.98 | 22.64 | 7.71 | 44.3259 |
| BTC | 2026-06-10 | 1.82 | cheap_vol_watch | 62.91 | 74.91 | 84.60 | -12.00 | -21.69 | 19.37 | 2.68 | 43.7359 |
| BTC | 2026-12-25 | 199.82 | cheap_vol_watch | 44.27 | 74.91 | 84.60 | -30.64 | -40.33 | 1.53 | -0.69 | 42.5459 |

## Interpretation

Positive IV premium means listed ATM IV is above recent realized volatility. This can point to vol-selling or event-premium candidates, but it still needs realized-vol forecasts, hedge PnL, option spreads, margin, and tail-risk controls.
