# Current OKX Liquidation Paper Gate

This subtracts assumed round-trip taker fees, current spread, and a simple visible-depth impact proxy from the 15m monitor-sample continuation label. It is a sizing gate, not a trade instruction.

| asset | action | size USD | gross bps | cost bps | net bps | near depth 5bps | depth usage | gate | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| JTO | long_liquidation_cascade_watch | 100 | 105.62 | 12.07 | 93.55 | 2208 | 0.0453 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| JTO | long_liquidation_cascade_watch | 250 | 105.62 | 12.75 | 92.87 | 2208 | 0.1132 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| JTO | long_liquidation_cascade_watch | 500 | 105.62 | 13.88 | 91.74 | 2208 | 0.2265 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| ONDO | short_liquidation_squeeze_watch | 100 | 74.52 | 12.95 | 61.57 | 13968 | 0.0072 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| ONDO | short_liquidation_squeeze_watch | 250 | 74.52 | 13.06 | 61.46 | 13968 | 0.0179 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| ONDO | short_liquidation_squeeze_watch | 500 | 74.52 | 13.24 | 61.28 | 13968 | 0.0358 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| ONDO | short_liquidation_squeeze_watch | 1000 | 74.52 | 13.60 | 60.92 | 13968 | 0.0716 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| ONDO | short_liquidation_squeeze_watch | 2500 | 74.52 | 14.67 | 59.85 | 13968 | 0.1790 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| H | short_liquidation_squeeze_watch | 100 | 61.16 | 13.07 | 48.09 | 767 | 0.1304 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| LTC | long_liquidation_cascade_watch | 100 | 25.99 | 12.42 | 13.57 | 26234 | 0.0038 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| LTC | long_liquidation_cascade_watch | 250 | 25.99 | 12.48 | 13.51 | 26234 | 0.0095 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| LTC | long_liquidation_cascade_watch | 500 | 25.99 | 12.57 | 13.41 | 26234 | 0.0191 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| LTC | long_liquidation_cascade_watch | 1000 | 25.99 | 12.76 | 13.22 | 26234 | 0.0381 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| LTC | long_liquidation_cascade_watch | 2500 | 25.99 | 13.34 | 12.65 | 26234 | 0.0953 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| LTC | long_liquidation_cascade_watch | 5000 | 25.99 | 14.29 | 11.70 | 26234 | 0.1906 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| JTO | long_liquidation_cascade_watch | 1000 | 105.62 | 16.14 | 89.47 | 2208 | 0.4529 | too_large_for_visible_depth | candidate size uses too much visible near-touch depth |
| JTO | long_liquidation_cascade_watch | 2500 | 105.62 | 21.61 | 84.00 | 2208 | 1.1324 | too_large_for_visible_depth | candidate size uses too much visible near-touch depth |
| JTO | long_liquidation_cascade_watch | 5000 | 105.62 | 21.61 | 84.00 | 2208 | 2.2647 | too_large_for_visible_depth | candidate size uses too much visible near-touch depth |
| ONDO | short_liquidation_squeeze_watch | 5000 | 74.52 | 16.46 | 58.06 | 13968 | 0.3580 | too_large_for_visible_depth | candidate size uses too much visible near-touch depth |
| H | short_liquidation_squeeze_watch | 250 | 61.16 | 15.03 | 46.13 | 767 | 0.3259 | too_large_for_visible_depth | candidate size uses too much visible near-touch depth |

## Interpretation

`small_paper_probe` means the current short-window label survives this rough fee/spread/depth check at the listed notional. This still omits real account fees, order-type choice, live spread changes, queue position, and stop logic.
