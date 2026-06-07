# Current OKX Liquidation Flow

This maps recent OKX USDT swap liquidation flow. Long liquidation means forced sell flow; short liquidation means forced buy flow.

| asset | action | obs | long liq USD | short liq USD | total liq USD | liq/vol | imbalance | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MRVL | long_liquidation_cascade_watch | 17 | 22227 | 0 | 22227 | 0.000556 | -1.000000 | 0.102508 |
| OPN | long_liquidation_cascade_watch | 82 | 21874 | 313 | 22187 | 0.000451 | -0.971786 | 0.089663 |
| XRP | short_liquidation_squeeze_watch | 4 | 0 | 69804 | 69804 | 0.000325 | 1.000000 | 0.087262 |
| BEAT | long_liquidation_cascade_watch | 34 | 69445 | 15445 | 84890 | 0.000247 | -0.636122 | 0.049243 |
| TON | short_liquidation_squeeze_watch | 2 | 0 | 7704 | 7704 | 0.000110 | 1.000000 | 0.040765 |
| ALLO | long_liquidation_cascade_watch | 33 | 23018 | 0 | 23018 | 0.000081 | -1.000000 | 0.039298 |
| JTO | long_liquidation_cascade_watch | 14 | 6414 | 0 | 6414 | 0.000105 | -1.000000 | 0.039068 |
| BSB | mixed_liquidation_flow_watch | 189 | 153446 | 94522 | 247968 | 0.000724 | -0.237626 | 0.034491 |
| ZEC | mixed_liquidation_flow_watch | 77 | 154739 | 89455 | 244194 | 0.000232 | -0.267343 | 0.021958 |
| HOME | short_liquidation_squeeze_watch | 6 | 262 | 1899 | 2162 | 0.000036 | 0.757186 | 0.015133 |
| WLD | mixed_liquidation_flow_watch | 42 | 36747 | 57379 | 94126 | 0.000185 | 0.219201 | 0.014843 |
| H | long_liquidation_cascade_watch | 7 | 1742 | 0 | 1742 | 0.000015 | -1.000000 | 0.012717 |
| LAB | long_liquidation_cascade_watch | 9 | 4635 | 0 | 4635 | 0.000012 | -1.000000 | 0.012550 |
| ETH | short_liquidation_squeeze_watch | 19 | 0 | 35450 | 35450 | 0.000005 | 1.000000 | 0.010301 |
| NEAR | long_liquidation_cascade_watch | 4 | 1343 | 0 | 1343 | 0.000010 | -1.000000 | 0.009837 |
| ONDO | long_liquidation_cascade_watch | 2 | 85 | 0 | 85 | 0.000002 | -1.000000 | 0.002814 |
| FIL | long_liquidation_cascade_watch | 1 | 73 | 0 | 73 | 0.000001 | -1.000000 | 0.002273 |
| SUI | short_liquidation_squeeze_watch | 1 | 0 | 58 | 58 | 0.000001 | 1.000000 | 0.001552 |
| HYPE | mixed_liquidation_flow_watch | 5 | 2317 | 2886 | 5203 | 0.000013 | 0.109472 | 0.001465 |
| BTC | short_liquidation_squeeze_watch | 1 | 0 | 255 | 255 | 0.000000 | 1.000000 | 0.000478 |
| ADA | short_liquidation_squeeze_watch | 1 | 0 | 12 | 12 | 0.000000 | 1.000000 | 0.000447 |
| XLM | long_liquidation_cascade_watch | 1 | 8 | 0 | 8 | 0.000000 | -1.000000 | 0.000315 |

## Interpretation

This is a live event-flow screen. It does not prove whether the right trade is continuation, reversal, or no trade. The next test is to label post-liquidation returns and join with funding, open interest, and order-book depth.
