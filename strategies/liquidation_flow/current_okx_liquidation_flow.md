# Current OKX Liquidation Flow

This maps recent OKX USDT swap liquidation flow. Long liquidation means forced sell flow; short liquidation means forced buy flow.

| asset | action | obs | long liq USD | short liq USD | total liq USD | liq/vol | imbalance | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H | long_liquidation_cascade_watch | 251 | 176861 | 15379 | 192240 | 0.002985 | -0.840002 | 0.242484 |
| PIPPIN | long_liquidation_cascade_watch | 142 | 86528 | 12002 | 98530 | 0.000463 | -0.756385 | 0.081236 |
| ZEC | short_liquidation_squeeze_watch | 44 | 324 | 140586 | 140910 | 0.000182 | 0.995405 | 0.069077 |
| MRVL | long_liquidation_cascade_watch | 9 | 25143 | 0 | 25143 | 0.000231 | -1.000000 | 0.066862 |
| ALLO | long_liquidation_cascade_watch | 90 | 64541 | 0 | 64541 | 0.000136 | -1.000000 | 0.056066 |
| WLD | mixed_liquidation_flow_watch | 214 | 283488 | 222762 | 506249 | 0.000971 | -0.119952 | 0.021319 |
| SOXL | long_liquidation_cascade_watch | 4 | 2252 | 0 | 2252 | 0.000040 | -1.000000 | 0.021141 |
| BEAT | long_liquidation_cascade_watch | 8 | 24982 | 4732 | 29715 | 0.000039 | -0.681490 | 0.018945 |
| ETH | short_liquidation_squeeze_watch | 4 | 3373 | 126892 | 130265 | 0.000015 | 0.948215 | 0.018536 |
| MU | long_liquidation_cascade_watch | 8 | 3847 | 0 | 3847 | 0.000013 | -1.000000 | 0.012688 |
| NEAR | long_liquidation_cascade_watch | 1 | 817 | 0 | 817 | 0.000005 | -1.000000 | 0.006727 |
| CBRS | long_liquidation_cascade_watch | 1 | 235 | 0 | 235 | 0.000004 | -1.000000 | 0.004992 |
| BSB | long_liquidation_cascade_watch | 2 | 488 | 0 | 488 | 0.000003 | -1.000000 | 0.004327 |
| SOL | short_liquidation_squeeze_watch | 3 | 0 | 1397 | 1397 | 0.000002 | 1.000000 | 0.004224 |
| HYPE | mixed_liquidation_flow_watch | 8 | 4496 | 2960 | 7456 | 0.000012 | -0.206083 | 0.002800 |
| CL | long_liquidation_cascade_watch | 1 | 91 | 0 | 91 | 0.000001 | -1.000000 | 0.001415 |
| BTC | short_liquidation_squeeze_watch | 1 | 0 | 1687 | 1687 | 0.000000 | 1.000000 | 0.001379 |
| BCH | short_liquidation_squeeze_watch | 1 | 0 | 23 | 23 | 0.000000 | 1.000000 | 0.000666 |
| DOGE | long_liquidation_cascade_watch | 1 | 1 | 0 | 1 | 0.000000 | -1.000000 | 0.000000 |

## Interpretation

This is a live event-flow screen. It does not prove whether the right trade is continuation, reversal, or no trade. The next test is to label post-liquidation returns and join with funding, open interest, and order-book depth.
