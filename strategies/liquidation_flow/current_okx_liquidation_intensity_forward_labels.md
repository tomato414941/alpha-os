# Current OKX Liquidation Intensity Forward Labels

This labels high liquidation/OI events from the current intensity screen. It is a continuation-versus-reversal check, not a trade instruction.

| asset | action | status | dir | intensity | cont 5m | cont 15m | cont 1h | rev 5m | rev 15m | rev 1h | label | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ALLO | short_liquidation_squeeze_watch | liquidation_oi_pressure_watch | 1 | 36.9253 | 0.024981 | 0.039872 |  | -0.024981 | -0.039872 |  | continuation_15m_supported_pending_1h | wait for ALLO short_liquidation_squeeze_watch 1h label, then add depth, fees, funding, and adverse-excursion checks |
| FIL | long_liquidation_cascade_watch | forced_flow_oi_shock_watch | -1 | 48.0291 | -0.005285 | -0.003964 |  | 0.005285 | 0.003964 |  | reversal_15m_supported_pending_1h | wait for FIL long_liquidation_cascade_watch 1h label, then add depth, fees, funding, and adverse-excursion checks |
| BNB | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | -1 | 17.3321 | -0.003543 | -0.003712 |  | 0.003543 | 0.003712 |  | reversal_15m_supported_pending_1h | wait for BNB long_liquidation_cascade_watch 1h label, then add depth, fees, funding, and adverse-excursion checks |
| ADA | long_liquidation_cascade_watch | forced_flow_oi_shock_watch | -1 | 28.2021 | -0.005415 | -0.003610 |  | 0.005415 | 0.003610 |  | reversal_15m_supported_pending_1h | wait for ADA long_liquidation_cascade_watch 1h label, then add depth, fees, funding, and adverse-excursion checks |
| PEPE | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | -1 | 22.9118 | -0.006173 | -0.002542 |  | 0.006173 | 0.002542 |  | reversal_15m_supported_pending_1h | wait for PEPE long_liquidation_cascade_watch 1h label, then add depth, fees, funding, and adverse-excursion checks |
| BCH | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | -1 | 25.0050 | -0.005874 | -0.002447 |  | 0.005874 | 0.002447 |  | reversal_15m_supported_pending_1h | wait for BCH long_liquidation_cascade_watch 1h label, then add depth, fees, funding, and adverse-excursion checks |
| PIPPIN | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | -1 | 176.3304 | -0.026403 |  |  | 0.026403 |  |  | label_pending | repeat PIPPIN long_liquidation_cascade_watch on a fresh liquidation/OI event before promotion |
| WLD | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | -1 | 62.7397 |  |  |  |  |  |  | label_pending | repeat WLD long_liquidation_cascade_watch on a fresh liquidation/OI event before promotion |
| SUI | long_liquidation_cascade_watch | forced_flow_oi_shock_watch | -1 | 48.3062 | -0.005206 |  |  | 0.005206 |  |  | label_pending | repeat SUI long_liquidation_cascade_watch on a fresh liquidation/OI event before promotion |
| SOL | long_liquidation_cascade_watch | forced_flow_oi_shock_watch | -1 | 39.5263 | 0.000764 |  |  | -0.000764 |  |  | label_pending | repeat SOL long_liquidation_cascade_watch on a fresh liquidation/OI event before promotion |
| DOGE | long_liquidation_cascade_watch | forced_flow_oi_shock_watch | -1 | 38.1857 | -0.000118 |  |  | 0.000118 |  |  | label_pending | repeat DOGE long_liquidation_cascade_watch on a fresh liquidation/OI event before promotion |
| ZEC | long_liquidation_cascade_watch | forced_flow_oi_shock_watch | -1 | 36.6909 |  |  |  |  |  |  | label_pending | repeat ZEC long_liquidation_cascade_watch on a fresh liquidation/OI event before promotion |
| ETH | long_liquidation_cascade_watch | forced_flow_oi_shock_watch | -1 | 33.7775 | -0.000867 |  |  | 0.000867 |  |  | label_pending | repeat ETH long_liquidation_cascade_watch on a fresh liquidation/OI event before promotion |
| BTC | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | -1 | 28.5504 | -0.001018 |  |  | 0.001018 |  |  | label_pending | repeat BTC long_liquidation_cascade_watch on a fresh liquidation/OI event before promotion |
| BSB | long_liquidation_cascade_watch | forced_flow_oi_shock_watch | -1 | 28.0384 |  |  |  |  |  |  | label_pending | repeat BSB long_liquidation_cascade_watch on a fresh liquidation/OI event before promotion |
| HYPE | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | -1 | 12.4040 | 0.000482 |  |  | -0.000482 |  |  | label_pending | repeat HYPE long_liquidation_cascade_watch on a fresh liquidation/OI event before promotion |
| H | mixed_liquidation_flow_watch | liquidation_oi_pressure_watch | 0 | 841.8515 |  |  |  |  |  |  | mixed_direction_unlabeled | do not promote H until mixed liquidation direction is separated |
| CL | mixed_liquidation_flow_watch | liquidation_oi_pressure_watch | 0 | 16.0538 |  |  |  |  |  |  | mixed_direction_unlabeled | do not promote CL until mixed liquidation direction is separated |

## Interpretation

Continuation means price moved in the forced-flow direction implied by the liquidation event. Reversal means price moved against that direction. These labels still exclude spread, fees, funding PnL, fill probability, and adverse-excursion stops.
