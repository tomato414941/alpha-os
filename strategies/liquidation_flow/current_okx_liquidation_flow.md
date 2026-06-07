# Current OKX Liquidation Flow

This maps recent OKX USDT swap liquidation flow. Long liquidation means forced sell flow; short liquidation means forced buy flow.

| asset | action | obs | long liq USD | short liq USD | total liq USD | liq/vol | imbalance | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MRVL | long_liquidation_cascade_watch | 17 | 22227 | 0 | 22227 | 0.000561 | -1.000000 | 0.102975 |
| BEAT | long_liquidation_cascade_watch | 84 | 154472 | 18538 | 173010 | 0.000503 | -0.785701 | 0.092315 |
| OPN | long_liquidation_cascade_watch | 84 | 22009 | 313 | 22322 | 0.000448 | -0.971956 | 0.089509 |
| XRP | short_liquidation_squeeze_watch | 3 | 0 | 69393 | 69393 | 0.000321 | 1.000000 | 0.086762 |
| TON | short_liquidation_squeeze_watch | 3 | 0 | 8084 | 8084 | 0.000115 | 1.000000 | 0.041915 |
| ALLO | long_liquidation_cascade_watch | 33 | 23018 | 0 | 23018 | 0.000079 | -1.000000 | 0.038791 |
| ZEC | mixed_liquidation_flow_watch | 81 | 206986 | 88907 | 295893 | 0.000281 | -0.399059 | 0.036582 |
| JTO | mixed_liquidation_flow_watch | 13 | 6374 | 2573 | 8948 | 0.000147 | -0.424787 | 0.020371 |
| ETH | short_liquidation_squeeze_watch | 53 | 0 | 105459 | 105459 | 0.000015 | 1.000000 | 0.019624 |
| WLD | mixed_liquidation_flow_watch | 38 | 32903 | 57379 | 90282 | 0.000174 | 0.271100 | 0.017727 |
| BSB | mixed_liquidation_flow_watch | 231 | 155534 | 124842 | 280376 | 0.000818 | -0.109466 | 0.017060 |
| H | long_liquidation_cascade_watch | 7 | 1742 | 0 | 1742 | 0.000016 | -1.000000 | 0.012830 |
| NEAR | long_liquidation_cascade_watch | 4 | 1343 | 0 | 1343 | 0.000010 | -1.000000 | 0.009852 |
| HOME | short_liquidation_squeeze_watch | 4 | 262 | 1107 | 1370 | 0.000023 | 0.616807 | 0.009347 |
| ONDO | long_liquidation_cascade_watch | 2 | 85 | 0 | 85 | 0.000002 | -1.000000 | 0.002815 |
| FIL | long_liquidation_cascade_watch | 1 | 73 | 0 | 73 | 0.000001 | -1.000000 | 0.002265 |
| LAB | long_liquidation_cascade_watch | 2 | 247 | 0 | 247 | 0.000001 | -1.000000 | 0.001866 |
| SUI | short_liquidation_squeeze_watch | 1 | 0 | 58 | 58 | 0.000001 | 1.000000 | 0.001551 |
| HYPE | mixed_liquidation_flow_watch | 4 | 2317 | 2017 | 4333 | 0.000011 | -0.069165 | 0.000826 |
| BTC | short_liquidation_squeeze_watch | 2 | 0 | 355 | 355 | 0.000000 | 1.000000 | 0.000597 |
| ADA | short_liquidation_squeeze_watch | 1 | 0 | 12 | 12 | 0.000000 | 1.000000 | 0.000446 |
| XLM | long_liquidation_cascade_watch | 1 | 8 | 0 | 8 | 0.000000 | -1.000000 | 0.000314 |

## Interpretation

This is a live event-flow screen. It does not prove whether the right trade is continuation, reversal, or no trade. The next test is to label post-liquidation returns and join with funding, open interest, and order-book depth.
