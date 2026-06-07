# Current OKX Liquidation Forward Labels

This labels liquidation-flow candidates with continuation returns. Positive continuation return means the forced-flow direction continued over that horizon.

| asset | action | dir | raw 15m | continuation 15m | raw 1h | continuation 1h |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ALLO | short_liquidation_squeeze_watch | 1 | 0.019775 | 0.019775 |  |  |
| H | short_liquidation_squeeze_watch | 1 | 0.013053 | 0.013053 |  |  |
| BEAT | short_liquidation_squeeze_watch | 1 | 0.006535 | 0.006535 |  |  |
| EDEN | long_liquidation_cascade_watch | -1 | -0.005715 | 0.005715 |  |  |
| LAB | short_liquidation_squeeze_watch | 1 | 0.004624 | 0.004624 |  |  |
| PEPE | short_liquidation_squeeze_watch | 1 | 0.003269 | 0.003269 | 0.003269 | 0.003269 |
| TON | short_liquidation_squeeze_watch | 1 | 0.002957 | 0.002957 | 0.002957 | 0.002957 |
| LTC | long_liquidation_cascade_watch | -1 | -0.002599 | 0.002599 |  |  |
| HYPE | long_liquidation_cascade_watch | -1 | -0.002376 | 0.002376 |  |  |
| DOGE | short_liquidation_squeeze_watch | 1 | 0.002012 | 0.002012 |  |  |
| PUMP | short_liquidation_squeeze_watch | 1 | 0.001991 | 0.001991 | 0.005309 | 0.005309 |
| XRP | short_liquidation_squeeze_watch | 1 | 0.001770 | 0.001770 | 0.003540 | 0.003540 |
| SOL | short_liquidation_squeeze_watch | 1 | 0.001699 | 0.001699 | 0.003707 | 0.003707 |
| NEAR | short_liquidation_squeeze_watch | 1 | 0.001465 | 0.001465 |  |  |
| ETH | short_liquidation_squeeze_watch | 1 | 0.001036 | 0.001036 |  |  |
| BTC | short_liquidation_squeeze_watch | 1 | 0.000920 | 0.000920 |  |  |
| BNB | short_liquidation_squeeze_watch | 1 | 0.000508 | 0.000508 | 0.000846 | 0.000846 |
| ONDO | short_liquidation_squeeze_watch | 1 | 0.000287 | 0.000287 |  |  |
| ZEC | short_liquidation_squeeze_watch | 1 | -0.000070 | -0.000070 |  |  |
| SUI | long_liquidation_cascade_watch | -1 | 0.000803 | -0.000803 | 0.000803 | -0.000803 |
| XLM | long_liquidation_cascade_watch | -1 | 0.001476 | -0.001476 |  |  |
| WLD | short_liquidation_squeeze_watch | 1 | -0.001606 | -0.001606 |  |  |
| OPN | long_liquidation_cascade_watch | -1 | 0.002292 | -0.002292 |  |  |
| HOME | long_liquidation_cascade_watch | -1 | 0.003196 | -0.003196 |  |  |
| JTO | long_liquidation_cascade_watch | -1 | 0.003874 | -0.003874 |  |  |

## Interpretation

This is price-only continuation labeling. It does not decide whether a liquidation event should be traded as continuation, reversal, or ignored without further regime and execution checks.
