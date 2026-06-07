# Current OKX Liquidation Forward Labels

This labels liquidation-flow candidates with continuation returns. Positive continuation return means the forced-flow direction continued over that horizon.

| asset | action | dir | raw 15m | continuation 15m | raw 1h | continuation 1h |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| WLD | short_liquidation_squeeze_watch | 1 | 0.027309 | 0.027309 |  |  |
| ALLO | short_liquidation_squeeze_watch | 1 | 0.019775 | 0.019775 |  |  |
| H | short_liquidation_squeeze_watch | 1 | 0.013053 | 0.013053 |  |  |
| LAB | short_liquidation_squeeze_watch | 1 | 0.004624 | 0.004624 |  |  |
| BEAT | short_liquidation_squeeze_watch | 1 | 0.004041 | 0.004041 |  |  |
| PEPE | short_liquidation_squeeze_watch | 1 | 0.003269 | 0.003269 | 0.005449 | 0.005449 |
| TON | short_liquidation_squeeze_watch | 1 | 0.002957 | 0.002957 | 0.002365 | 0.002365 |
| LTC | long_liquidation_cascade_watch | -1 | -0.002599 | 0.002599 |  |  |
| HYPE | long_liquidation_cascade_watch | -1 | -0.002376 | 0.002376 |  |  |
| BTC | short_liquidation_squeeze_watch | 1 | 0.002046 | 0.002046 |  |  |
| DOGE | short_liquidation_squeeze_watch | 1 | 0.002012 | 0.002012 |  |  |
| ONDO | short_liquidation_squeeze_watch | 1 | 0.002006 | 0.002006 |  |  |
| PUMP | short_liquidation_squeeze_watch | 1 | 0.001991 | 0.001991 | 0.005309 | 0.005309 |
| XRP | short_liquidation_squeeze_watch | 1 | 0.001770 | 0.001770 | 0.005310 | 0.005310 |
| SOL | short_liquidation_squeeze_watch | 1 | 0.001699 | 0.001699 | 0.005406 | 0.005406 |
| ETH | short_liquidation_squeeze_watch | 1 | 0.001110 | 0.001110 |  |  |
| BNB | short_liquidation_squeeze_watch | 1 | 0.000508 | 0.000508 | 0.003047 | 0.003047 |
| JTO | long_liquidation_cascade_watch | -1 | -0.000323 | 0.000323 |  |  |
| SUI | long_liquidation_cascade_watch | -1 | 0.000803 | -0.000803 | 0.001071 | -0.001071 |
| XLM | long_liquidation_cascade_watch | -1 | 0.001476 | -0.001476 |  |  |
| OPN | long_liquidation_cascade_watch | -1 | 0.002292 | -0.002292 |  |  |
| EDEN | long_liquidation_cascade_watch | -1 | 0.003239 | -0.003239 |  |  |
| ZEC | short_liquidation_squeeze_watch | 1 | -0.006303 | -0.006303 |  |  |
| HOME | long_liquidation_cascade_watch | -1 | 0.007351 | -0.007351 |  |  |
| NEAR | short_liquidation_squeeze_watch | 1 | -0.009277 | -0.009277 |  |  |

## Interpretation

This is price-only continuation labeling. It does not decide whether a liquidation event should be traded as continuation, reversal, or ignored without further regime and execution checks.
