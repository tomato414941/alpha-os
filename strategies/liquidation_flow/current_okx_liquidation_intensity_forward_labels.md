# Current OKX Liquidation Intensity Forward Labels

This labels high liquidation/OI events from the current intensity screen. It is a continuation-versus-reversal check, not a trade instruction.

| asset | action | status | dir | intensity | cont 5m | cont 15m | cont 1h | rev 5m | rev 15m | rev 1h | label | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| BEAT | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | -1 | 12.8974 | -0.000046 | -0.015777 |  | 0.000046 | 0.015777 |  | reversal_15m_supported_pending_1h | wait for BEAT long_liquidation_cascade_watch 1h label, then add depth, fees, funding, and adverse-excursion checks |
| ALLO | long_liquidation_cascade_watch | forced_flow_oi_shock_watch | -1 | 86.8724 | -0.010963 | -0.012570 |  | 0.010963 | 0.012570 |  | reversal_15m_supported_pending_1h | wait for ALLO long_liquidation_cascade_watch 1h label, then add depth, fees, funding, and adverse-excursion checks |
| H | long_liquidation_cascade_watch | forced_flow_oi_shock_watch | -1 | 217.2056 | 0.024534 |  |  | -0.024534 |  |  | label_pending | repeat H long_liquidation_cascade_watch on a fresh liquidation/OI event before promotion |
| PIPPIN | long_liquidation_cascade_watch | forced_flow_oi_shock_watch | -1 | 157.0726 | 0.001838 |  |  | -0.001838 |  |  | label_pending | repeat PIPPIN long_liquidation_cascade_watch on a fresh liquidation/OI event before promotion |
| ZEC | short_liquidation_squeeze_watch | forced_flow_oi_shock_watch | 1 | 31.3056 | -0.006087 |  |  | 0.006087 |  |  | label_pending | repeat ZEC short_liquidation_squeeze_watch on a fresh liquidation/OI event before promotion |
| MRVL | long_liquidation_cascade_watch | forced_flow_oi_shock_watch | -1 | 29.3188 | -0.000685 |  |  | 0.000685 |  |  | label_pending | repeat MRVL long_liquidation_cascade_watch on a fresh liquidation/OI event before promotion |
| WLD | mixed_liquidation_flow_watch | liquidation_oi_pressure_watch | 0 | 120.8562 |  |  |  |  |  |  | mixed_direction_unlabeled | do not promote WLD until mixed liquidation direction is separated |

## Interpretation

Continuation means price moved in the forced-flow direction implied by the liquidation event. Reversal means price moved against that direction. These labels still exclude spread, fees, funding PnL, fill probability, and adverse-excursion stops.
