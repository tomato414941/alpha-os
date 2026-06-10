# Current OKX Liquidation Flow

This maps recent OKX USDT swap liquidation flow. Long liquidation means forced sell flow; short liquidation means forced buy flow.

| asset | action | obs | long liq USD | short liq USD | total liq USD | liq/vol | imbalance | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SUI | long_liquidation_cascade_watch | 26 | 80911 | 0 | 80911 | 0.001221 | -1.000000 | 0.171525 |
| SOL | long_liquidation_cascade_watch | 102 | 574464 | 0 | 574464 | 0.000828 | -1.000000 | 0.165757 |
| BTC | long_liquidation_cascade_watch | 406 | 3558949 | 0 | 3558949 | 0.000478 | -1.000000 | 0.143298 |
| DOGE | long_liquidation_cascade_watch | 120 | 208127 | 0 | 208127 | 0.000707 | -1.000000 | 0.141364 |
| FIL | long_liquidation_cascade_watch | 27 | 46251 | 0 | 46251 | 0.000906 | -1.000000 | 0.140412 |
| H | mixed_liquidation_flow_watch | 807 | 221051 | 106145 | 327197 | 0.004769 | -0.351183 | 0.133747 |
| BNB | long_liquidation_cascade_watch | 11 | 46037 | 0 | 46037 | 0.000718 | -1.000000 | 0.124984 |
| ADA | long_liquidation_cascade_watch | 12 | 42134 | 0 | 42134 | 0.000680 | -1.000000 | 0.120624 |
| ETH | long_liquidation_cascade_watch | 427 | 2866313 | 49824 | 2916137 | 0.000363 | -0.965829 | 0.119029 |
| BCH | long_liquidation_cascade_watch | 19 | 40809 | 0 | 40809 | 0.000425 | -1.000000 | 0.095073 |
| PIPPIN | long_liquidation_cascade_watch | 191 | 84041 | 14591 | 98632 | 0.000505 | -0.704132 | 0.079059 |
| PEPE | long_liquidation_cascade_watch | 34 | 27452 | 0 | 27452 | 0.000292 | -1.000000 | 0.075813 |
| ZEC | long_liquidation_cascade_watch | 57 | 162557 | 17380 | 179937 | 0.000248 | -0.806817 | 0.066744 |
| WLD | long_liquidation_cascade_watch | 83 | 164925 | 39269 | 204194 | 0.000412 | -0.615371 | 0.066302 |
| XRP | long_liquidation_cascade_watch | 18 | 27969 | 0 | 27969 | 0.000160 | -1.000000 | 0.056291 |
| HYPE | long_liquidation_cascade_watch | 25 | 51089 | 1696 | 52785 | 0.000096 | -0.935728 | 0.043273 |
| BSB | long_liquidation_cascade_watch | 25 | 12091 | 0 | 12091 | 0.000081 | -1.000000 | 0.036668 |
| NEAR | long_liquidation_cascade_watch | 17 | 10104 | 0 | 10104 | 0.000072 | -1.000000 | 0.033943 |
| SNDK | long_liquidation_cascade_watch | 2 | 6915 | 0 | 6915 | 0.000066 | -1.000000 | 0.031257 |
| ALLO | short_liquidation_squeeze_watch | 49 | 3628 | 22795 | 26423 | 0.000054 | 0.725393 | 0.023539 |
| LAB | long_liquidation_cascade_watch | 19 | 4004 | 269 | 4273 | 0.000045 | -0.874135 | 0.021332 |
| MU | long_liquidation_cascade_watch | 2 | 4004 | 0 | 4004 | 0.000014 | -1.000000 | 0.013619 |
| BEAT | short_liquidation_squeeze_watch | 14 | 2231 | 11004 | 13234 | 0.000019 | 0.662887 | 0.012014 |
| XAU | long_liquidation_cascade_watch | 1 | 2541 | 0 | 2541 | 0.000011 | -1.000000 | 0.011375 |
| XAG | long_liquidation_cascade_watch | 3 | 910 | 0 | 910 | 0.000007 | -1.000000 | 0.007907 |

## Interpretation

This is a live event-flow screen. It does not prove whether the right trade is continuation, reversal, or no trade. The next test is to label post-liquidation returns and join with funding, open interest, and order-book depth.
