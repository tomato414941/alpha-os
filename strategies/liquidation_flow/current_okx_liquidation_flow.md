# Current OKX Liquidation Flow

This maps recent OKX USDT swap liquidation flow. Long liquidation means forced sell flow; short liquidation means forced buy flow.

| asset | action | obs | long liq USD | short liq USD | total liq USD | liq/vol | imbalance | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MRVL | short_liquidation_squeeze_watch | 8 | 0 | 117073 | 117073 | 0.001263 | 1.000000 | 0.180109 |
| BCH | short_liquidation_squeeze_watch | 6 | 0 | 46269 | 46269 | 0.000527 | 1.000000 | 0.107128 |
| SNDK | short_liquidation_squeeze_watch | 9 | 0 | 33920 | 33920 | 0.000343 | 1.000000 | 0.083947 |
| ETH | short_liquidation_squeeze_watch | 140 | 44063 | 1215859 | 1259922 | 0.000144 | 0.930054 | 0.068054 |
| BSB | short_liquidation_squeeze_watch | 21 | 0 | 43377 | 43377 | 0.000124 | 1.000000 | 0.051692 |
| WLD | long_liquidation_cascade_watch | 44 | 95426 | 26530 | 121956 | 0.000296 | -0.564928 | 0.049469 |
| BTC | short_liquidation_squeeze_watch | 117 | 0 | 614341 | 614341 | 0.000065 | 1.000000 | 0.046572 |
| ZEC | short_liquidation_squeeze_watch | 64 | 6394 | 78263 | 84657 | 0.000104 | 0.848938 | 0.042713 |
| BNB | short_liquidation_squeeze_watch | 1 | 0 | 5291 | 5291 | 0.000062 | 1.000000 | 0.029364 |
| MU | long_liquidation_cascade_watch | 7 | 13116 | 313 | 13428 | 0.000048 | -0.953448 | 0.027357 |
| XRP | short_liquidation_squeeze_watch | 5 | 0 | 9249 | 9249 | 0.000046 | 1.000000 | 0.026781 |
| XAU | long_liquidation_cascade_watch | 1 | 9769 | 0 | 9769 | 0.000042 | -1.000000 | 0.025872 |
| DOGE | short_liquidation_squeeze_watch | 1 | 0 | 6224 | 6224 | 0.000018 | 1.000000 | 0.016165 |
| ALLO | mixed_liquidation_flow_watch | 137 | 59852 | 37379 | 97231 | 0.000191 | -0.231131 | 0.015938 |
| XAG | long_liquidation_cascade_watch | 1 | 2450 | 0 | 2450 | 0.000019 | -1.000000 | 0.014641 |
| TON | short_liquidation_squeeze_watch | 2 | 93 | 1116 | 1209 | 0.000025 | 0.846869 | 0.013137 |
| HYPE | long_liquidation_cascade_watch | 8 | 9516 | 2599 | 12115 | 0.000021 | -0.570949 | 0.010731 |
| SOL | short_liquidation_squeeze_watch | 6 | 0 | 4274 | 4274 | 0.000006 | 1.000000 | 0.008646 |
| CL | short_liquidation_squeeze_watch | 1 | 0 | 231 | 231 | 0.000001 | 1.000000 | 0.002687 |
| ONDO | short_liquidation_squeeze_watch | 1 | 0 | 92 | 92 | 0.000002 | 1.000000 | 0.002686 |
| NEAR | short_liquidation_squeeze_watch | 1 | 0 | 100 | 100 | 0.000001 | 1.000000 | 0.001537 |
| HOME | long_liquidation_cascade_watch | 1 | 30 | 0 | 30 | 0.000001 | -1.000000 | 0.001080 |
| FIL | long_liquidation_cascade_watch | 1 | 26 | 0 | 26 | 0.000000 | -1.000000 | 0.000915 |
| ADA | short_liquidation_squeeze_watch | 1 | 0 | 26 | 26 | 0.000000 | 1.000000 | 0.000857 |
| BEAT | mixed_liquidation_flow_watch | 253 | 138131 | 140147 | 278278 | 0.000327 | 0.007245 | 0.000713 |

## Interpretation

This is a live event-flow screen. It does not prove whether the right trade is continuation, reversal, or no trade. The next test is to label post-liquidation returns and join with funding, open interest, and order-book depth.
