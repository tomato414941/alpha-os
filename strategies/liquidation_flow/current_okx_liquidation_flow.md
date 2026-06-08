# Current OKX Liquidation Flow

This maps recent OKX USDT swap liquidation flow. Long liquidation means forced sell flow; short liquidation means forced buy flow.

| asset | action | obs | long liq USD | short liq USD | total liq USD | liq/vol | imbalance | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XAU | long_liquidation_cascade_watch | 183 | 1250437 | 2179 | 1252616 | 0.007966 | -0.996520 | 0.542360 |
| XAG | long_liquidation_cascade_watch | 55 | 219499 | 6605 | 226105 | 0.003298 | -0.941572 | 0.289536 |
| SUI | long_liquidation_cascade_watch | 9 | 31704 | 0 | 31704 | 0.000366 | -1.000000 | 0.086141 |
| BEAT | short_liquidation_squeeze_watch | 163 | 0 | 177381 | 177381 | 0.000254 | 1.000000 | 0.083681 |
| MU | long_liquidation_cascade_watch | 12 | 37995 | 0 | 37995 | 0.000253 | -1.000000 | 0.072847 |
| HYPE | short_liquidation_squeeze_watch | 32 | 0 | 76460 | 76460 | 0.000161 | 1.000000 | 0.061870 |
| DOGE | long_liquidation_cascade_watch | 15 | 58987 | 0 | 58987 | 0.000162 | -1.000000 | 0.060743 |
| BSB | short_liquidation_squeeze_watch | 94 | 1402 | 46358 | 47760 | 0.000141 | 0.941290 | 0.052363 |
| PIPPIN | long_liquidation_cascade_watch | 58 | 12959 | 2672 | 15631 | 0.000242 | -0.658067 | 0.042925 |
| WLD | long_liquidation_cascade_watch | 73 | 67514 | 14209 | 81723 | 0.000179 | -0.652272 | 0.042818 |
| ALLO | short_liquidation_squeeze_watch | 49 | 3919 | 31027 | 34946 | 0.000138 | 0.775715 | 0.041350 |
| MRVL | long_liquidation_cascade_watch | 2 | 4872 | 0 | 4872 | 0.000096 | -1.000000 | 0.036077 |
| NEAR | short_liquidation_squeeze_watch | 8 | 333 | 12198 | 12531 | 0.000085 | 0.946882 | 0.035707 |
| LAB | short_liquidation_squeeze_watch | 41 | 0 | 13391 | 13391 | 0.000073 | 1.000000 | 0.035147 |
| PEPE | long_liquidation_cascade_watch | 2 | 10493 | 0 | 10493 | 0.000076 | -1.000000 | 0.035068 |
| SOL | long_liquidation_cascade_watch | 11 | 41505 | 0 | 41505 | 0.000052 | -1.000000 | 0.033143 |
| XRP | long_liquidation_cascade_watch | 5 | 14661 | 0 | 14661 | 0.000060 | -1.000000 | 0.032236 |
| BTC | short_liquidation_squeeze_watch | 150 | 133537 | 524081 | 657618 | 0.000070 | 0.593877 | 0.028862 |
| ETH | mixed_liquidation_flow_watch | 200 | 220537 | 620420 | 840956 | 0.000092 | 0.475510 | 0.027067 |
| HOME | mixed_liquidation_flow_watch | 14 | 2267 | 5089 | 7355 | 0.000091 | 0.383643 | 0.014151 |
| JTO | long_liquidation_cascade_watch | 12 | 972 | 28 | 1000 | 0.000015 | -0.944093 | 0.011058 |
| SNDK | long_liquidation_cascade_watch | 3 | 758 | 0 | 758 | 0.000014 | -1.000000 | 0.010956 |
| ZEC | mixed_liquidation_flow_watch | 21 | 6500 | 13097 | 19597 | 0.000020 | 0.336683 | 0.006392 |
| TON | short_liquidation_squeeze_watch | 3 | 168 | 612 | 781 | 0.000014 | 0.568555 | 0.006096 |
| CL | short_liquidation_squeeze_watch | 1 | 0 | 235 | 235 | 0.000002 | 1.000000 | 0.003417 |

## Interpretation

This is a live event-flow screen. It does not prove whether the right trade is continuation, reversal, or no trade. The next test is to label post-liquidation returns and join with funding, open interest, and order-book depth.
