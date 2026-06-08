# Current OKX Liquidation Flow

This maps recent OKX USDT swap liquidation flow. Long liquidation means forced sell flow; short liquidation means forced buy flow.

| asset | action | obs | long liq USD | short liq USD | total liq USD | liq/vol | imbalance | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MRVL | short_liquidation_squeeze_watch | 8 | 0 | 117073 | 117073 | 0.001237 | 1.000000 | 0.178242 |
| SNDK | short_liquidation_squeeze_watch | 17 | 0 | 74706 | 74706 | 0.000737 | 1.000000 | 0.132284 |
| ETH | short_liquidation_squeeze_watch | 189 | 0 | 2979699 | 2979699 | 0.000337 | 1.000000 | 0.118838 |
| BCH | short_liquidation_squeeze_watch | 7 | 0 | 46478 | 46478 | 0.000520 | 1.000000 | 0.106429 |
| BTC | short_liquidation_squeeze_watch | 190 | 0 | 1895640 | 1895640 | 0.000199 | 1.000000 | 0.088471 |
| PEPE | short_liquidation_squeeze_watch | 4 | 0 | 35228 | 35228 | 0.000282 | 1.000000 | 0.076340 |
| BSB | short_liquidation_squeeze_watch | 27 | 0 | 41657 | 41657 | 0.000120 | 1.000000 | 0.050602 |
| BEAT | short_liquidation_squeeze_watch | 125 | 15755 | 112429 | 128184 | 0.000151 | 0.754186 | 0.047273 |
| ZEC | short_liquidation_squeeze_watch | 68 | 6394 | 79696 | 86091 | 0.000107 | 0.851454 | 0.043437 |
| BNB | short_liquidation_squeeze_watch | 2 | 0 | 7111 | 7111 | 0.000083 | 1.000000 | 0.035163 |
| XRP | short_liquidation_squeeze_watch | 9 | 0 | 12491 | 12491 | 0.000061 | 1.000000 | 0.031922 |
| TON | short_liquidation_squeeze_watch | 3 | 0 | 2549 | 2549 | 0.000052 | 1.000000 | 0.024635 |
| PIPPIN | short_liquidation_squeeze_watch | 39 | 1342 | 7716 | 9058 | 0.000077 | 0.703748 | 0.024470 |
| DOGE | short_liquidation_squeeze_watch | 9 | 0 | 9979 | 9979 | 0.000029 | 1.000000 | 0.021465 |
| WLD | mixed_liquidation_flow_watch | 55 | 95749 | 66049 | 161798 | 0.000390 | -0.183565 | 0.018887 |
| SOL | short_liquidation_squeeze_watch | 16 | 0 | 12796 | 12796 | 0.000017 | 1.000000 | 0.016795 |
| ALLO | mixed_liquidation_flow_watch | 136 | 59785 | 37379 | 97165 | 0.000189 | -0.230603 | 0.015824 |
| ADA | short_liquidation_squeeze_watch | 3 | 0 | 1000 | 1000 | 0.000014 | 1.000000 | 0.011298 |
| HYPE | long_liquidation_cascade_watch | 8 | 9516 | 2599 | 12115 | 0.000021 | -0.570949 | 0.010684 |
| MU | short_liquidation_squeeze_watch | 5 | 0 | 2363 | 2363 | 0.000008 | 1.000000 | 0.009716 |
| SUI | short_liquidation_squeeze_watch | 3 | 0 | 871 | 871 | 0.000010 | 1.000000 | 0.009490 |
| ONDO | short_liquidation_squeeze_watch | 1 | 0 | 92 | 92 | 0.000002 | 1.000000 | 0.002682 |
| CL | short_liquidation_squeeze_watch | 1 | 0 | 231 | 231 | 0.000001 | 1.000000 | 0.002680 |
| NEAR | short_liquidation_squeeze_watch | 1 | 0 | 100 | 100 | 0.000001 | 1.000000 | 0.001535 |
| HOME | long_liquidation_cascade_watch | 1 | 30 | 0 | 30 | 0.000001 | -1.000000 | 0.001082 |

## Interpretation

This is a live event-flow screen. It does not prove whether the right trade is continuation, reversal, or no trade. The next test is to label post-liquidation returns and join with funding, open interest, and order-book depth.
