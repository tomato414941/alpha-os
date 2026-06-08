# Current OFI Execution Survival

This joins the book-depth execution-cost sweep to current L2 imbalance assets. Rows are execution-survival probes, not standalone trading strategies.

| asset | status | action | mode | score | maker net | pressure | spread | depth | net15 | net1h | next probe |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ETH | short_horizon_maker_probe_only | paper_short | maker_or_internalized | 201.9019 | 0.0957 | 0.0219 | 0.5934 | 11568191.61 | 61.7839 | -27.7985 | label ETH OFI at 5m/15m only; do not hold to 1h without a separate rule |
| SUI | short_horizon_maker_probe_only | paper_short | maker_or_internalized | 158.3900 | 0.0957 | 0.1561 | 1.2273 | 70814.73 | 36.9174 | -34.8434 | label SUI OFI at 5m/15m only; do not hold to 1h without a separate rule |
| BNB | short_horizon_maker_probe_only | paper_short | maker_or_internalized | 146.5600 | 0.0957 | 0.0742 | 1.9211 | 68200.59 | 24.9236 | -61.8084 | label BNB OFI at 5m/15m only; do not hold to 1h without a separate rule |
| ZEC | current_l2_label_missing_or_negative | paper_short | maker_or_internalized | 96.9129 | 0.0957 | 0.3296 | 1.1749 | 68103.13 | -47.7389 | 163.8154 | collect a fresh ZEC L2 label before using OFI as an alpha feature |
| HYPE | current_l2_label_missing_or_negative | paper_short | maker_or_internalized | 96.4656 | 0.0957 | 0.6768 | 0.3152 | 182454.29 | -253.4329 | 37.4205 | collect a fresh HYPE L2 label before using OFI as an alpha feature |
| ENA | current_l2_label_missing_or_negative | paper_short | maker_or_internalized | 66.5405 | 0.0957 | 0.2986 | 2.0573 | 29494.24 | -69.6160 | 39.3049 | collect a fresh ENA L2 label before using OFI as an alpha feature |
| XRP | current_l2_label_missing_or_negative | paper_short | maker_or_internalized | 63.4739 | 0.0957 | 0.2080 | 0.8511 | 593877.05 | -84.7982 | -41.3086 | collect a fresh XRP L2 label before using OFI as an alpha feature |
| LTC | current_l2_label_missing_or_negative | paper_short | maker_or_internalized | 63.1399 | 0.0957 | 0.2640 | 2.3798 | 47453.55 | -25.6757 | 34.6893 | collect a fresh LTC L2 label before using OFI as an alpha feature |
| VVV | current_l2_label_missing_or_negative | paper_short | maker_or_internalized | 55.5341 | 0.0957 | 0.3713 | 2.4896 | 2085.48 | -177.7902 | 37.1516 | collect a fresh VVV L2 label before using OFI as an alpha feature |
| ONDO | current_l2_label_missing_or_negative | paper_short | maker_or_internalized | 48.3621 | 0.0957 | 0.2230 | 3.5366 | 13633.76 | -72.5334 | -28.5265 | collect a fresh ONDO L2 label before using OFI as an alpha feature |
| JTO | execution_world_blocks_ofi | paper_short | maker_or_internalized | -36.9143 | 0.0957 | 0.8916 | 7.0095 | 1875.52 | -37.5801 | 94.6843 | reject JTO OFI for current size or wait for tighter spread and deeper book |
| WLD | execution_world_blocks_ofi | paper_short | maker_or_internalized | -68.0419 | 0.0957 | 0.4699 | 7.4499 | 17455.16 | -1.5542 | -185.3087 | reject WLD OFI for current size or wait for tighter spread and deeper book |
