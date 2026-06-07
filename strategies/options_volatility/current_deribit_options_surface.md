# Current Deribit Options Surface

This compresses public Deribit BTC/ETH option summaries into ATM IV, simple 5% OTM skew, and adjacent-expiry term structure. It is a volatility-surface exploration probe, not a trade instruction.

| currency | expiry | dte | atm iv | skew iv | term iv spread | spread pct | oi | volume USD | action | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| BTC | 2026-06-08 | 0.63 | 64.68 | 20.42 | -9.26 | 0.2593 | 4889 | 889676 | put_skew_watch | 48.0600 |
| BTC | 2026-06-12 | 4.63 | 64.15 | 15.46 | 10.39 | 0.0519 | 30582 | 1200670 | put_skew_watch | 46.7012 |
| ETH | 2026-06-12 | 4.63 | 80.84 | 10.33 | 10.91 | 0.0447 | 136986 | 192091 | front_vol_premium_watch | 42.4809 |
| ETH | 2026-06-08 | 0.63 | 84.72 | 23.44 | -2.71 | 0.1316 | 52768 | 289041 | put_skew_watch | 38.7802 |
| BTC | 2026-06-09 | 1.63 | 73.94 | 20.32 | 4.42 | 0.1028 | 3174 | 935784 | put_skew_watch | 38.4273 |
| BTC | 2026-06-10 | 2.63 | 69.52 | 19.06 | 2.96 | 0.0548 | 961 | 487311 | put_skew_watch | 33.5411 |
| ETH | 2026-06-09 | 1.63 | 87.43 | 14.77 | 3.72 | 0.0783 | 33526 | 172778 | put_skew_watch | 31.8163 |
| BTC | 2026-06-26 | 18.63 | 49.70 | 6.69 | 4.50 | 0.0283 | 143587 | 2238139 | put_skew_watch | 27.1405 |
| ETH | 2026-06-26 | 18.63 | 66.08 | 4.72 | 5.60 | 0.0331 | 904860 | 207576 | front_vol_premium_watch | 27.1275 |
| BTC | 2026-06-19 | 11.63 | 53.76 | 8.96 | 4.06 | 0.0321 | 15386 | 503135 | put_skew_watch | 26.9046 |
| BTC | 2026-06-11 | 3.63 | 66.56 | 15.13 | 2.41 | 0.0761 | 86 | 70961 | put_skew_watch | 26.5874 |
| ETH | 2026-06-19 | 11.63 | 69.93 | 6.37 | 3.85 | 0.0374 | 54356 | 132190 | put_skew_watch | 23.8517 |
| ETH | 2026-06-10 | 2.63 | 83.71 | 11.64 | 1.59 | 0.0488 | 4673 | 67365 | put_skew_watch | 23.2204 |
| ETH | 2026-06-11 | 3.63 | 82.12 | 10.19 | 1.28 | 0.0565 | 240 | 12897 | put_skew_watch | 19.1296 |
| BTC | 2026-07-31 | 53.63 | 45.20 | 3.74 | 1.25 | 0.0204 | 37128 | 1378335 | surface_context | 16.9082 |
| ETH | 2026-07-31 | 53.63 | 60.48 | 2.33 | 1.03 | 0.0213 | 138940 | 100968 | surface_context | 14.4945 |
| BTC | 2026-09-25 | 109.63 | 44.39 | 1.88 | -0.57 | 0.0236 | 76368 | 1077581 | surface_context | 13.8882 |
| BTC | 2026-12-25 | 200.63 | 44.96 | 1.34 | -0.70 | 0.0214 | 80354 | 1329938 | surface_context | 13.7261 |
| ETH | 2026-12-25 | 200.63 | 59.92 | 0.58 | -1.30 | 0.0265 | 394404 | 31906 | surface_context | 13.2268 |
| BTC | 2026-08-28 | 81.63 | 43.95 | 2.87 | -0.44 | 0.0249 | 6592 | 368979 | surface_context | 13.0862 |

## Interpretation

Large positive term spread means the nearer expiry has richer ATM IV than the next expiry. Positive skew means the 5% OTM put proxy is richer than the 5% OTM call proxy. This still needs realized-vol baselines, option execution costs, margin, and hedging rules.
