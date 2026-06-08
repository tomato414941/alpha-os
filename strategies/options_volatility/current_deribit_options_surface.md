# Current Deribit Options Surface

This compresses public Deribit BTC/ETH option summaries into ATM IV, simple 5% OTM skew, and adjacent-expiry term structure. It is a volatility-surface exploration probe, not a trade instruction.

| currency | expiry | dte | atm iv | skew iv | term iv spread | spread pct | oi | volume USD | action | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| BTC | 2026-06-12 | 3.87 | 56.19 | 16.38 | 7.37 | 0.0730 | 31697 | 1971540 | put_skew_watch | 41.7699 |
| BTC | 2026-06-09 | 0.87 | 61.29 | 24.00 | 2.57 | 0.2090 | 4659 | 954546 | put_skew_watch | 38.3702 |
| ETH | 2026-06-12 | 3.87 | 71.63 | 15.61 | 4.92 | 0.0554 | 145492 | 768674 | put_skew_watch | 36.3879 |
| BTC | 2026-06-10 | 1.87 | 58.72 | 19.60 | 1.07 | 0.0653 | 1812 | 781458 | put_skew_watch | 30.7606 |
| ETH | 2026-06-09 | 0.87 | 71.00 | 18.66 | -0.22 | 0.0940 | 45986 | 307952 | put_skew_watch | 29.0632 |
| BTC | 2026-06-11 | 2.87 | 57.65 | 17.87 | 1.46 | 0.0880 | 571 | 291591 | put_skew_watch | 28.8362 |
| ETH | 2026-06-10 | 1.87 | 71.22 | 15.88 | -0.88 | 0.0597 | 14514 | 240070 | put_skew_watch | 27.0627 |
| ETH | 2026-06-26 | 17.87 | 63.49 | 5.66 | 4.63 | 0.0309 | 905512 | 717363 | put_skew_watch | 26.6708 |
| BTC | 2026-06-19 | 10.87 | 48.82 | 9.21 | 2.72 | 0.0473 | 16364 | 1271627 | put_skew_watch | 24.8736 |
| ETH | 2026-06-19 | 10.87 | 66.71 | 7.60 | 3.22 | 0.0407 | 59676 | 495926 | put_skew_watch | 24.4298 |
| BTC | 2026-06-26 | 17.87 | 46.10 | 7.26 | 2.52 | 0.0354 | 144546 | 4908106 | put_skew_watch | 24.0801 |
| ETH | 2026-06-11 | 2.87 | 72.10 | 13.38 | 0.47 | 0.0676 | 6793 | 203940 | put_skew_watch | 23.3265 |
| ETH | 2026-09-25 | 108.87 | 57.78 | 1.73 | -1.78 | 0.0122 | 293646 | 553239 | surface_context | 16.4763 |
| BTC | 2026-09-25 | 108.87 | 42.91 | 1.98 | -1.56 | 0.0185 | 76286 | 2944506 | surface_context | 16.4145 |
| ETH | 2026-07-31 | 52.87 | 58.86 | 2.70 | 1.20 | 0.0162 | 143192 | 1047081 | surface_context | 16.2436 |
| BTC | 2026-07-31 | 52.87 | 43.58 | 4.15 | 0.63 | 0.0136 | 38094 | 1522521 | surface_context | 16.1463 |
| ETH | 2026-12-25 | 199.87 | 59.56 | 1.20 | -1.19 | 0.0161 | 396913 | 383742 | surface_context | 14.7305 |
| BTC | 2026-12-25 | 199.87 | 44.47 | 1.06 | -0.60 | 0.0136 | 80134 | 6125112 | surface_context | 13.9237 |
| BTC | 2026-08-28 | 80.87 | 42.95 | 2.93 | 0.04 | 0.0180 | 6860 | 656831 | surface_context | 12.6278 |
| BTC | 2027-03-26 | 290.87 | 45.07 | 0.85 |  | 0.0195 | 18367 | 2071406 | surface_context | 11.3914 |

## Interpretation

Large positive term spread means the nearer expiry has richer ATM IV than the next expiry. Positive skew means the 5% OTM put proxy is richer than the 5% OTM call proxy. This still needs realized-vol baselines, option execution costs, margin, and hedging rules.
