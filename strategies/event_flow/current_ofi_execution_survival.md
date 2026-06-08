# Current OFI Execution Survival

This joins the book-depth execution-cost sweep to current L2 imbalance assets. Rows are execution-survival probes, not standalone trading strategies.

| asset | status | action | mode | score | maker net | pressure | spread | depth | net15 | net1h | next probe |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BTC | short_horizon_maker_probe_only | paper_short | maker_or_internalized | 206.3180 | 0.0957 | 0.2654 | 0.1558 | 4641067.36 | 33.0256 | -36.5748 | label BTC OFI at 5m/15m only; do not hold to 1h without a separate rule |
| HYPE | current_l2_label_missing_or_negative | paper_short | maker_or_internalized | 97.1293 | 0.0957 | 0.7006 | 0.4618 | 153346.85 | -253.4329 | 37.4205 | collect a fresh HYPE L2 label before using OFI as an alpha feature |
| ZEC | current_l2_label_missing_or_negative | paper_short | maker_or_internalized | 90.9251 | 0.0957 | 0.2223 | 1.1301 | 89408.82 | -47.7389 | 163.8154 | collect a fresh ZEC L2 label before using OFI as an alpha feature |
| AVAX | current_l2_label_missing_or_negative | paper_short | maker_or_internalized | 74.5589 | 0.0957 | 0.1500 | 1.2133 | 55285.70 | -57.0873 | 107.6602 | collect a fresh AVAX L2 label before using OFI as an alpha feature |
| ONDO | current_l2_label_missing_or_negative | paper_short | maker_or_internalized | 59.0235 | 0.0957 | 0.3157 | 1.6283 | 10713.46 | -72.5334 | -28.5265 | collect a fresh ONDO L2 label before using OFI as an alpha feature |
| ADA | current_l2_label_missing_or_negative | paper_short | maker_or_internalized | 57.8580 | 0.0957 | 0.0271 | 2.7306 | 55649.00 | -89.7191 | 83.7335 | collect a fresh ADA L2 label before using OFI as an alpha feature |
| LIT | execution_world_blocks_ofi | paper_short | maker_or_internalized | -64.1197 | 0.0957 | 0.6176 | 8.1279 | 2197.74 | 22.9454 | -148.0922 | reject LIT OFI for current size or wait for tighter spread and deeper book |
| JTO | execution_world_blocks_ofi | paper_short | maker_or_internalized | -92.0194 | 0.0957 | 0.1854 | 12.1414 | 2249.24 | -37.5801 | 94.6843 | reject JTO OFI for current size or wait for tighter spread and deeper book |
