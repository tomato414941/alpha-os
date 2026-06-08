# Current OKX Liquidation Flow

This maps recent OKX USDT swap liquidation flow. Long liquidation means forced sell flow; short liquidation means forced buy flow.

| asset | action | obs | long liq USD | short liq USD | total liq USD | liq/vol | imbalance | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MU | long_liquidation_cascade_watch | 6 | 201391 | 0 | 201391 | 0.002439 | -1.000000 | 0.261965 |
| XAU | short_liquidation_squeeze_watch | 15 | 0 | 51243 | 51243 | 0.000580 | 1.000000 | 0.113375 |
| BEAT | long_liquidation_cascade_watch | 202 | 187365 | 51093 | 238458 | 0.000517 | -0.571470 | 0.069886 |
| HOME | short_liquidation_squeeze_watch | 7 | 0 | 9107 | 9107 | 0.000150 | 1.000000 | 0.048525 |
| ETH | short_liquidation_squeeze_watch | 68 | 0 | 232837 | 232837 | 0.000026 | 1.000000 | 0.027547 |
| BTC | short_liquidation_squeeze_watch | 57 | 0 | 197164 | 197164 | 0.000022 | 1.000000 | 0.024575 |
| ALLO | mixed_liquidation_flow_watch | 41 | 8328 | 22375 | 30703 | 0.000119 | 0.457491 | 0.022360 |
| WLD | mixed_liquidation_flow_watch | 28 | 8731 | 25243 | 33973 | 0.000065 | 0.486036 | 0.017776 |
| ZEC | short_liquidation_squeeze_watch | 28 | 4950 | 25463 | 30413 | 0.000026 | 0.674462 | 0.015408 |
| BSB | short_liquidation_squeeze_watch | 51 | 4380 | 13403 | 17782 | 0.000050 | 0.507429 | 0.015287 |
| CL | long_liquidation_cascade_watch | 1 | 782 | 0 | 782 | 0.000013 | -1.000000 | 0.010593 |
| LAB | long_liquidation_cascade_watch | 7 | 823 | 0 | 823 | 0.000003 | -1.000000 | 0.005422 |
| OPN | mixed_liquidation_flow_watch | 9 | 711 | 372 | 1083 | 0.000022 | -0.313268 | 0.004452 |
| SUI | mixed_liquidation_flow_watch | 2 | 126 | 353 | 479 | 0.000005 | 0.472369 | 0.002840 |
| HYPE | long_liquidation_cascade_watch | 1 | 23 | 0 | 23 | 0.000000 | -1.000000 | 0.000329 |
| PEPE | long_liquidation_cascade_watch | 1 | 3 | 0 | 3 | 0.000000 | -1.000000 | 0.000060 |

## Interpretation

This is a live event-flow screen. It does not prove whether the right trade is continuation, reversal, or no trade. The next test is to label post-liquidation returns and join with funding, open interest, and order-book depth.
