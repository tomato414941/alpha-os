# Current OKX Liquidation Intensity Forward Labels

This labels high liquidation/OI events from the current intensity screen. It is a continuation-versus-reversal check, not a trade instruction.

| asset | action | status | dir | intensity | cont 5m | cont 15m | cont 1h | rev 5m | rev 15m | rev 1h | label | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| MRVL | short_liquidation_squeeze_watch | forced_flow_oi_shock_watch | 1 | 111.0820 | -0.008264 | -0.011901 |  | 0.008264 | 0.011901 |  | reversal_15m_supported_pending_1h | wait for MRVL short_liquidation_squeeze_watch 1h label, then add depth, fees, funding, and adverse-excursion checks |
| BEAT | short_liquidation_squeeze_watch | forced_flow_oi_shock_watch | 1 | 37.1195 | 0.000456 | 0.007813 |  | -0.000456 | -0.007813 |  | continuation_15m_supported_pending_1h | wait for BEAT short_liquidation_squeeze_watch 1h label, then add depth, fees, funding, and adverse-excursion checks |
| SNDK | short_liquidation_squeeze_watch | forced_flow_oi_shock_watch | 1 | 74.6006 |  |  |  |  |  |  | label_pending | repeat SNDK short_liquidation_squeeze_watch on a fresh liquidation/OI event before promotion |
| BSB | short_liquidation_squeeze_watch | forced_flow_oi_shock_watch | 1 | 63.0405 | -0.000587 |  |  | 0.000587 |  |  | label_pending | repeat BSB short_liquidation_squeeze_watch on a fresh liquidation/OI event before promotion |
| ETH | short_liquidation_squeeze_watch | forced_flow_oi_shock_watch | 1 | 33.3949 |  |  |  |  |  |  | label_pending | repeat ETH short_liquidation_squeeze_watch on a fresh liquidation/OI event before promotion |
| BCH | short_liquidation_squeeze_watch | forced_flow_oi_shock_watch | 1 | 27.0887 | -0.001899 |  |  | 0.001899 |  |  | label_pending | repeat BCH short_liquidation_squeeze_watch on a fresh liquidation/OI event before promotion |
| PEPE | short_liquidation_squeeze_watch | forced_flow_oi_shock_watch | 1 | 26.7761 |  |  |  |  |  |  | label_pending | repeat PEPE short_liquidation_squeeze_watch on a fresh liquidation/OI event before promotion |
| ZEC | short_liquidation_squeeze_watch | liquidation_oi_pressure_watch | 1 | 20.0676 |  |  |  |  |  |  | label_pending | repeat ZEC short_liquidation_squeeze_watch on a fresh liquidation/OI event before promotion |
| BTC | short_liquidation_squeeze_watch | liquidation_oi_pressure_watch | 1 | 18.5485 |  |  |  |  |  |  | label_pending | repeat BTC short_liquidation_squeeze_watch on a fresh liquidation/OI event before promotion |
| ALLO | mixed_liquidation_flow_watch | liquidation_oi_pressure_watch | 0 | 104.6465 |  |  |  |  |  |  | mixed_direction_unlabeled | do not promote ALLO until mixed liquidation direction is separated |
| WLD | mixed_liquidation_flow_watch | liquidation_oi_pressure_watch | 0 | 53.5466 |  |  |  |  |  |  | mixed_direction_unlabeled | do not promote WLD until mixed liquidation direction is separated |

## Interpretation

Continuation means price moved in the forced-flow direction implied by the liquidation event. Reversal means price moved against that direction. These labels still exclude spread, fees, funding PnL, fill probability, and adverse-excursion stops.
