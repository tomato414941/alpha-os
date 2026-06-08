# Current OKX Liquidation Flow

This maps recent OKX USDT swap liquidation flow. Long liquidation means forced sell flow; short liquidation means forced buy flow.

| asset | action | obs | long liq USD | short liq USD | total liq USD | liq/vol | imbalance | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H | long_liquidation_cascade_watch | 295 | 88078 | 27111 | 115189 | 0.001885 | -0.529284 | 0.116305 |
| SPCX | long_liquidation_cascade_watch | 18 | 32489 | 0 | 32489 | 0.000413 | -1.000000 | 0.091689 |
| LAB | long_liquidation_cascade_watch | 39 | 26830 | 0 | 26830 | 0.000269 | -1.000000 | 0.072665 |
| PEPE | long_liquidation_cascade_watch | 2 | 5899 | 0 | 5899 | 0.000063 | -1.000000 | 0.030048 |
| BEAT | long_liquidation_cascade_watch | 30 | 33442 | 2406 | 35848 | 0.000052 | -0.865761 | 0.028474 |
| WLD | mixed_liquidation_flow_watch | 51 | 89895 | 46736 | 136631 | 0.000274 | -0.315874 | 0.026851 |
| HYPE | long_liquidation_cascade_watch | 14 | 16514 | 280 | 16795 | 0.000028 | -0.966615 | 0.021772 |
| ETH | short_liquidation_squeeze_watch | 86 | 87909 | 279285 | 367194 | 0.000046 | 0.521184 | 0.019579 |
| BTC | long_liquidation_cascade_watch | 32 | 162402 | 26595 | 188997 | 0.000025 | -0.718567 | 0.019140 |
| PIPPIN | mixed_liquidation_flow_watch | 13 | 1938 | 5195 | 7132 | 0.000036 | 0.456657 | 0.010498 |
| MRVL | long_liquidation_cascade_watch | 2 | 961 | 0 | 961 | 0.000009 | -1.000000 | 0.008924 |
| NEAR | long_liquidation_cascade_watch | 2 | 979 | 0 | 979 | 0.000007 | -1.000000 | 0.007840 |
| BSB | mixed_liquidation_flow_watch | 10 | 3987 | 2129 | 6116 | 0.000037 | -0.303700 | 0.007039 |
| ADA | long_liquidation_cascade_watch | 1 | 314 | 0 | 314 | 0.000005 | -1.000000 | 0.005640 |
| ZEC | mixed_liquidation_flow_watch | 33 | 17486 | 22962 | 40448 | 0.000054 | 0.135375 | 0.004591 |
| FIL | long_liquidation_cascade_watch | 1 | 141 | 0 | 141 | 0.000003 | -1.000000 | 0.003531 |
| SOL | long_liquidation_cascade_watch | 2 | 234 | 0 | 234 | 0.000000 | -1.000000 | 0.001379 |

## Interpretation

This is a live event-flow screen. It does not prove whether the right trade is continuation, reversal, or no trade. The next test is to label post-liquidation returns and join with funding, open interest, and order-book depth.
