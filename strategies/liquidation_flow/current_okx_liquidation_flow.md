# Current OKX Liquidation Flow

This maps recent OKX USDT swap liquidation flow. Long liquidation means forced sell flow; short liquidation means forced buy flow.

| asset | action | obs | long liq USD | short liq USD | total liq USD | liq/vol | imbalance | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TON | short_liquidation_squeeze_watch | 2 | 0 | 35445 | 35445 | 0.000707 | 1.000000 | 0.120948 |
| ZEC | long_liquidation_cascade_watch | 109 | 264429 | 3899 | 268327 | 0.000313 | -0.970942 | 0.093323 |
| ONDO | long_liquidation_cascade_watch | 4 | 15566 | 0 | 15566 | 0.000327 | -1.000000 | 0.075822 |
| ALLO | mixed_liquidation_flow_watch | 583 | 272234 | 134191 | 406425 | 0.000929 | -0.339652 | 0.058073 |
| MU | short_liquidation_squeeze_watch | 2 | 0 | 24448 | 24448 | 0.000127 | 1.000000 | 0.049411 |
| PIPPIN | long_liquidation_cascade_watch | 30 | 16411 | 1138 | 17548 | 0.000171 | -0.870308 | 0.048359 |
| ETH | long_liquidation_cascade_watch | 264 | 1232239 | 332467 | 1564706 | 0.000182 | -0.575043 | 0.048053 |
| PEPE | short_liquidation_squeeze_watch | 6 | 306 | 8149 | 8455 | 0.000068 | 0.927603 | 0.030070 |
| HYPE | long_liquidation_cascade_watch | 7 | 22822 | 1027 | 23849 | 0.000050 | -0.913851 | 0.028412 |
| WLD | long_liquidation_cascade_watch | 29 | 28063 | 5281 | 33344 | 0.000081 | -0.683248 | 0.027828 |
| BTC | mixed_liquidation_flow_watch | 409 | 1301181 | 731689 | 2032870 | 0.000223 | -0.280142 | 0.026370 |
| XAU | short_liquidation_squeeze_watch | 2 | 0 | 5380 | 5380 | 0.000026 | 1.000000 | 0.018966 |
| SUI | long_liquidation_cascade_watch | 3 | 2821 | 199 | 3021 | 0.000037 | -0.867958 | 0.018426 |
| BSB | mixed_liquidation_flow_watch | 37 | 16671 | 5626 | 22296 | 0.000060 | -0.495379 | 0.016718 |
| BEAT | mixed_liquidation_flow_watch | 132 | 80850 | 51220 | 132070 | 0.000164 | -0.224351 | 0.014726 |
| SOL | mixed_liquidation_flow_watch | 13 | 18973 | 38424 | 57397 | 0.000081 | 0.338877 | 0.014473 |
| H | short_liquidation_squeeze_watch | 1 | 0 | 1408 | 1408 | 0.000019 | 1.000000 | 0.013791 |
| DOGE | mixed_liquidation_flow_watch | 8 | 4169 | 9059 | 13229 | 0.000040 | 0.369646 | 0.009636 |
| CL | long_liquidation_cascade_watch | 2 | 924 | 0 | 924 | 0.000006 | -1.000000 | 0.007012 |
| BCH | long_liquidation_cascade_watch | 6 | 882 | 216 | 1098 | 0.000014 | -0.605924 | 0.006782 |
| LAB | short_liquidation_squeeze_watch | 1 | 0 | 643 | 643 | 0.000004 | 1.000000 | 0.005627 |
| NEAR | mixed_liquidation_flow_watch | 6 | 2021 | 1036 | 3057 | 0.000019 | -0.322001 | 0.004844 |
| SNDK | short_liquidation_squeeze_watch | 1 | 0 | 167 | 167 | 0.000002 | 1.000000 | 0.003447 |
| FIL | short_liquidation_squeeze_watch | 2 | 15 | 156 | 171 | 0.000003 | 0.820559 | 0.003023 |
| XAG | short_liquidation_squeeze_watch | 1 | 0 | 163 | 163 | 0.000001 | 1.000000 | 0.002661 |

## Interpretation

This is a live event-flow screen. It does not prove whether the right trade is continuation, reversal, or no trade. The next test is to label post-liquidation returns and join with funding, open interest, and order-book depth.
