# Current OKX Liquidation Intensity Forward Labels

This labels high liquidation/OI events from the current intensity screen. It is a continuation-versus-reversal check, not a trade instruction.

| asset | action | status | dir | intensity | cont 5m | cont 15m | cont 1h | rev 5m | rev 15m | rev 1h | label | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| MU | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | -1 | 17.6489 | -0.003731 | -0.006315 | -0.003892 | 0.003731 | 0.006315 | 0.003892 | reversal_15m_1h_supported | gate MU long_liquidation_cascade_watch with OKX depth, fees, funding, fill, and stop assumptions |
| SUI | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | -1 | 22.2527 | -0.003155 | -0.006310 | -0.020850 | 0.003155 | 0.006310 | 0.020850 | reversal_15m_1h_supported | gate SUI long_liquidation_cascade_watch with OKX depth, fees, funding, fill, and stop assumptions |
| PEPE | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | -1 | 12.3356 | -0.001821 | -0.004372 | -0.010200 | 0.001821 | 0.004372 | 0.010200 | reversal_15m_1h_supported | gate PEPE long_liquidation_cascade_watch with OKX depth, fees, funding, fill, and stop assumptions |
| DOGE | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | -1 | 15.3939 | -0.001421 | -0.002724 | -0.008173 | 0.001421 | 0.002724 | 0.008173 | reversal_15m_1h_supported | gate DOGE long_liquidation_cascade_watch with OKX depth, fees, funding, fill, and stop assumptions |
| XAU | long_liquidation_cascade_watch | forced_flow_oi_shock_watch | -1 | 130.5896 | 0.001067 | 0.000232 | 0.001856 | -0.001067 | -0.000232 | -0.001856 | continuation_15m_1h_supported | gate XAU long_liquidation_cascade_watch with OKX depth, fees, funding, fill, and stop assumptions |
| LAB | short_liquidation_squeeze_watch | liquidation_oi_pressure_watch | 1 | 13.3575 | -0.027222 | -0.037560 |  | 0.027222 | 0.037560 |  | reversal_15m_supported_pending_1h | wait for LAB short_liquidation_squeeze_watch 1h label, then add depth, fees, funding, and adverse-excursion checks |
| PIPPIN | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | -1 | 41.1213 | -0.022527 | -0.028022 |  | 0.022527 | 0.028022 |  | reversal_15m_supported_pending_1h | wait for PIPPIN long_liquidation_cascade_watch 1h label, then add depth, fees, funding, and adverse-excursion checks |
| ALLO | short_liquidation_squeeze_watch | forced_flow_oi_shock_watch | 1 | 47.1519 | 0.013881 | 0.010141 |  | -0.013881 | -0.010141 |  | continuation_15m_supported_pending_1h | wait for ALLO short_liquidation_squeeze_watch 1h label, then add depth, fees, funding, and adverse-excursion checks |
| BEAT | short_liquidation_squeeze_watch | forced_flow_oi_shock_watch | 1 | 53.7012 | 0.007076 | -0.009756 |  | -0.007076 | 0.009756 |  | reversal_15m_supported_pending_1h | wait for BEAT short_liquidation_squeeze_watch 1h label, then add depth, fees, funding, and adverse-excursion checks |
| XAG | long_liquidation_cascade_watch | forced_flow_oi_shock_watch | -1 | 106.7256 | -0.000000 | 0.005195 |  | 0.000000 | -0.005195 |  | continuation_15m_supported_pending_1h | wait for XAG long_liquidation_cascade_watch 1h label, then add depth, fees, funding, and adverse-excursion checks |
| BSB | short_liquidation_squeeze_watch | forced_flow_oi_shock_watch | 1 | 74.0609 | 0.011612 | -0.003981 |  | -0.011612 | 0.003981 |  | reversal_15m_supported_pending_1h | wait for BSB short_liquidation_squeeze_watch 1h label, then add depth, fees, funding, and adverse-excursion checks |
| WLD | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | -1 | 29.8900 | 0.001674 | -0.002929 |  | -0.001674 | 0.002929 |  | reversal_15m_supported_pending_1h | wait for WLD long_liquidation_cascade_watch 1h label, then add depth, fees, funding, and adverse-excursion checks |
| HYPE | short_liquidation_squeeze_watch | liquidation_oi_pressure_watch | 1 | 15.3030 | 0.000159 | -0.000478 |  | -0.000159 | 0.000478 |  | reversal_15m_supported_pending_1h | wait for HYPE short_liquidation_squeeze_watch 1h label, then add depth, fees, funding, and adverse-excursion checks |
| ETH | mixed_liquidation_flow_watch | liquidation_oi_pressure_watch | 0 | 14.1268 |  |  |  |  |  |  | mixed_direction_unlabeled | do not promote ETH until mixed liquidation direction is separated |

## Interpretation

Continuation means price moved in the forced-flow direction implied by the liquidation event. Reversal means price moved against that direction. These labels still exclude spread, fees, funding PnL, fill probability, and adverse-excursion stops.
