# Current OKX Liquidation Intensity Forward Labels

This labels high liquidation/OI events from the current intensity screen. It is a continuation-versus-reversal check, not a trade instruction.

| asset | action | status | dir | intensity | cont 5m | cont 15m | cont 1h | rev 5m | rev 15m | rev 1h | label | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| MRVL | short_liquidation_squeeze_watch | forced_flow_oi_shock_watch | 1 | 108.6918 | -0.008264 |  |  | 0.008264 |  |  | label_pending | repeat MRVL short_liquidation_squeeze_watch on a fresh liquidation/OI event before promotion |
| BSB | short_liquidation_squeeze_watch | forced_flow_oi_shock_watch | 1 | 66.3291 |  |  |  |  |  |  | label_pending | repeat BSB short_liquidation_squeeze_watch on a fresh liquidation/OI event before promotion |
| WLD | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | -1 | 43.0924 |  |  |  |  |  |  | label_pending | repeat WLD long_liquidation_cascade_watch on a fresh liquidation/OI event before promotion |
| SNDK | short_liquidation_squeeze_watch | forced_flow_oi_shock_watch | 1 | 38.0260 |  |  |  |  |  |  | label_pending | repeat SNDK short_liquidation_squeeze_watch on a fresh liquidation/OI event before promotion |
| BCH | short_liquidation_squeeze_watch | forced_flow_oi_shock_watch | 1 | 27.2775 |  |  |  |  |  |  | label_pending | repeat BCH short_liquidation_squeeze_watch on a fresh liquidation/OI event before promotion |
| ZEC | short_liquidation_squeeze_watch | liquidation_oi_pressure_watch | 1 | 20.0091 |  |  |  |  |  |  | label_pending | repeat ZEC short_liquidation_squeeze_watch on a fresh liquidation/OI event before promotion |
| ETH | short_liquidation_squeeze_watch | liquidation_oi_pressure_watch | 1 | 18.5279 |  |  |  |  |  |  | label_pending | repeat ETH short_liquidation_squeeze_watch on a fresh liquidation/OI event before promotion |
| ALLO | mixed_liquidation_flow_watch | liquidation_oi_pressure_watch | 0 | 104.3465 |  |  |  |  |  |  | mixed_direction_unlabeled | do not promote ALLO until mixed liquidation direction is separated |
| BEAT | mixed_liquidation_flow_watch | liquidation_oi_pressure_watch | 0 | 72.2306 |  |  |  |  |  |  | mixed_direction_unlabeled | do not promote BEAT until mixed liquidation direction is separated |

## Interpretation

Continuation means price moved in the forced-flow direction implied by the liquidation event. Reversal means price moved against that direction. These labels still exclude spread, fees, funding PnL, fill probability, and adverse-excursion stops.
