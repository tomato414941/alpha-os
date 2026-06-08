# Current OKX Liquidation Intensity Forward Labels

This labels high liquidation/OI events from the current intensity screen. It is a continuation-versus-reversal check, not a trade instruction.

| asset | action | status | dir | intensity | cont 5m | cont 15m | cont 1h | rev 5m | rev 15m | rev 1h | label | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| H | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | -1 | 289.2080 |  |  |  |  |  |  | label_pending | repeat H long_liquidation_cascade_watch on a fresh liquidation/OI event before promotion |
| LAB | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | -1 | 23.1292 |  |  |  |  |  |  | label_pending | repeat LAB long_liquidation_cascade_watch on a fresh liquidation/OI event before promotion |
| SPCX | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | -1 | 18.2951 | -0.004036 |  |  | 0.004036 |  |  | label_pending | repeat SPCX long_liquidation_cascade_watch on a fresh liquidation/OI event before promotion |
| BEAT | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | -1 | 15.2104 | 0.024413 |  |  | -0.024413 |  |  | label_pending | repeat BEAT long_liquidation_cascade_watch on a fresh liquidation/OI event before promotion |
| WLD | mixed_liquidation_flow_watch | liquidation_oi_pressure_watch | 0 | 40.5367 |  |  |  |  |  |  | mixed_direction_unlabeled | do not promote WLD until mixed liquidation direction is separated |
| ZEC | mixed_liquidation_flow_watch | liquidation_oi_pressure_watch | 0 | 11.6897 |  |  |  |  |  |  | mixed_direction_unlabeled | do not promote ZEC until mixed liquidation direction is separated |

## Interpretation

Continuation means price moved in the forced-flow direction implied by the liquidation event. Reversal means price moved against that direction. These labels still exclude spread, fees, funding PnL, fill probability, and adverse-excursion stops.
