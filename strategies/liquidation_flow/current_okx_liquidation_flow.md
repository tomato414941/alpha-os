# Current OKX Liquidation Flow

This maps recent OKX USDT swap liquidation flow. Long liquidation means forced sell flow; short liquidation means forced buy flow.

| asset | action | obs | long liq USD | short liq USD | total liq USD | liq/vol | imbalance | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ZEC | short_liquidation_squeeze_watch | 440 | 112707 | 1193458 | 1306165 | 0.001213 | 0.827424 | 0.176258 |
| WLD | short_liquidation_squeeze_watch | 257 | 65082 | 432606 | 497689 | 0.001053 | 0.738462 | 0.136540 |
| JTO | long_liquidation_cascade_watch | 69 | 49065 | 4117 | 53183 | 0.000935 | -0.845163 | 0.122133 |
| BEAT | short_liquidation_squeeze_watch | 96 | 0 | 135450 | 135450 | 0.000474 | 1.000000 | 0.111679 |
| BSB | mixed_liquidation_flow_watch | 504 | 152970 | 295347 | 448316 | 0.002350 | 0.317581 | 0.087000 |
| HOME | long_liquidation_cascade_watch | 80 | 32025 | 6585 | 38610 | 0.000674 | -0.658901 | 0.078440 |
| ONDO | short_liquidation_squeeze_watch | 2 | 0 | 10345 | 10345 | 0.000273 | 1.000000 | 0.066295 |
| OPN | long_liquidation_cascade_watch | 39 | 17755 | 1283 | 19038 | 0.000288 | -0.865203 | 0.062830 |
| BTC | short_liquidation_squeeze_watch | 160 | 263236 | 927918 | 1191155 | 0.000186 | 0.558015 | 0.046283 |
| PEPE | short_liquidation_squeeze_watch | 6 | 0 | 12015 | 12015 | 0.000097 | 1.000000 | 0.040081 |
| SOL | short_liquidation_squeeze_watch | 20 | 0 | 50061 | 50061 | 0.000068 | 1.000000 | 0.038801 |
| H | short_liquidation_squeeze_watch | 16 | 0 | 9431 | 9431 | 0.000080 | 1.000000 | 0.035465 |
| NEAR | short_liquidation_squeeze_watch | 15 | 0 | 8912 | 8912 | 0.000080 | 1.000000 | 0.035300 |
| LTC | long_liquidation_cascade_watch | 1 | 4187 | 0 | 4187 | 0.000093 | -1.000000 | 0.035014 |
| ETH | short_liquidation_squeeze_watch | 199 | 186240 | 605599 | 791839 | 0.000113 | 0.529601 | 0.033152 |
| DOGE | short_liquidation_squeeze_watch | 20 | 1966 | 23291 | 25258 | 0.000076 | 0.844290 | 0.032341 |
| ALLO | short_liquidation_squeeze_watch | 40 | 0 | 15633 | 15633 | 0.000049 | 1.000000 | 0.029414 |
| XLM | long_liquidation_cascade_watch | 3 | 3343 | 0 | 3343 | 0.000043 | -1.000000 | 0.023146 |
| SUI | long_liquidation_cascade_watch | 3 | 2159 | 92 | 2251 | 0.000028 | -0.918302 | 0.016429 |
| BNB | short_liquidation_squeeze_watch | 2 | 0 | 1044 | 1044 | 0.000015 | 1.000000 | 0.011495 |
| LAB | short_liquidation_squeeze_watch | 8 | 158 | 3996 | 4154 | 0.000010 | 0.923851 | 0.010778 |
| TON | short_liquidation_squeeze_watch | 2 | 0 | 937 | 937 | 0.000012 | 1.000000 | 0.010503 |
| XRP | short_liquidation_squeeze_watch | 4 | 0 | 1047 | 1047 | 0.000005 | 1.000000 | 0.006508 |
| HYPE | long_liquidation_cascade_watch | 3 | 1653 | 306 | 1959 | 0.000005 | -0.687097 | 0.005001 |
| PUMP | short_liquidation_squeeze_watch | 1 | 0 | 37 | 37 | 0.000001 | 1.000000 | 0.001539 |

## Interpretation

This is a live event-flow screen. It does not prove whether the right trade is continuation, reversal, or no trade. The next test is to label post-liquidation returns and join with funding, open interest, and order-book depth.
