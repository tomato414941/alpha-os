# Current OKX Liquidation Intensity Forward Labels

This labels high liquidation/OI events from the current intensity screen. It is a continuation-versus-reversal check, not a trade instruction.

| asset | action | status | dir | intensity | cont 5m | cont 15m | cont 1h | rev 5m | rev 15m | rev 1h | label | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| TON | short_liquidation_squeeze_watch | forced_flow_oi_shock_watch | 1 | 28.9596 | 0.000000 | -0.006901 |  | -0.000000 | 0.006901 |  | reversal_15m_supported_pending_1h | wait for TON short_liquidation_squeeze_watch 1h label, then add depth, fees, funding, and adverse-excursion checks |
| ONDO | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | -1 | 17.5642 | -0.003617 | 0.002226 |  | 0.003617 | -0.002226 |  | continuation_15m_supported_pending_1h | wait for ONDO long_liquidation_cascade_watch 1h label, then add depth, fees, funding, and adverse-excursion checks |
| ZEC | long_liquidation_cascade_watch | forced_flow_oi_shock_watch | -1 | 51.2893 | 0.008411 |  |  | -0.008411 |  |  | label_pending | repeat ZEC long_liquidation_cascade_watch on a fresh liquidation/OI event before promotion |
| PIPPIN | long_liquidation_cascade_watch | forced_flow_oi_shock_watch | -1 | 46.4037 |  |  |  |  |  |  | label_pending | repeat PIPPIN long_liquidation_cascade_watch on a fresh liquidation/OI event before promotion |
| ETH | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | -1 | 20.5823 |  |  |  |  |  |  | label_pending | repeat ETH long_liquidation_cascade_watch on a fresh liquidation/OI event before promotion |
| WLD | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | -1 | 16.0922 |  |  |  |  |  |  | label_pending | repeat WLD long_liquidation_cascade_watch on a fresh liquidation/OI event before promotion |
| MU | short_liquidation_squeeze_watch | liquidation_oi_pressure_watch | 1 | 13.1475 | 0.000946 |  |  | -0.000946 |  |  | label_pending | repeat MU short_liquidation_squeeze_watch on a fresh liquidation/OI event before promotion |
| ALLO | mixed_liquidation_flow_watch | liquidation_oi_pressure_watch | 0 | 420.6765 |  |  |  |  |  |  | mixed_direction_unlabeled | do not promote ALLO until mixed liquidation direction is separated |
| BEAT | mixed_liquidation_flow_watch | liquidation_oi_pressure_watch | 0 | 38.3625 |  |  |  |  |  |  | mixed_direction_unlabeled | do not promote BEAT until mixed liquidation direction is separated |
| BSB | mixed_liquidation_flow_watch | liquidation_oi_pressure_watch | 0 | 34.4804 |  |  |  |  |  |  | mixed_direction_unlabeled | do not promote BSB until mixed liquidation direction is separated |
| BTC | mixed_liquidation_flow_watch | liquidation_oi_pressure_watch | 0 | 18.0971 |  |  |  |  |  |  | mixed_direction_unlabeled | do not promote BTC until mixed liquidation direction is separated |

## Interpretation

Continuation means price moved in the forced-flow direction implied by the liquidation event. Reversal means price moved against that direction. These labels still exclude spread, fees, funding PnL, fill probability, and adverse-excursion stops.
