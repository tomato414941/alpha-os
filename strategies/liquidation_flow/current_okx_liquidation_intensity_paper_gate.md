# Current OKX Liquidation Intensity Paper Gate

This applies a rough OKX spread, taker-fee, and visible-depth haircut to liquidation-intensity forward labels. It is not a trade instruction.

| asset | action | label | side | size USD | label bps | cost bps | net bps | depth 10bps USD | usage | gate | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| BEAT | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 100 | 157.77 | 10.29 | 147.48 | 15545 | 0.0064 | small_paper_probe_pending_1h | wait for BEAT 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| BEAT | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 250 | 157.77 | 10.39 | 147.38 | 15545 | 0.0161 | small_paper_probe_pending_1h | wait for BEAT 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| BEAT | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 500 | 157.77 | 10.55 | 147.22 | 15545 | 0.0322 | small_paper_probe_pending_1h | wait for BEAT 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| BEAT | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 1000 | 157.77 | 10.87 | 146.90 | 15545 | 0.0643 | small_paper_probe_pending_1h | wait for BEAT 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| ALLO | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 100 | 125.70 | 10.37 | 115.33 | 8713 | 0.0115 | small_paper_probe_pending_1h | wait for ALLO 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| ALLO | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 250 | 125.70 | 10.54 | 115.16 | 8713 | 0.0287 | small_paper_probe_pending_1h | wait for ALLO 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| ALLO | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 500 | 125.70 | 10.83 | 114.87 | 8713 | 0.0574 | small_paper_probe_pending_1h | wait for ALLO 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| ALLO | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 1000 | 125.70 | 11.40 | 114.30 | 8713 | 0.1148 | small_paper_probe_pending_1h | wait for ALLO 1h label, then paper-check with fill, funding, stop, and repeat-event logs |

## Interpretation

`small_paper_probe` means a label with 1h support still survives this rough gate. `small_paper_probe_pending_1h` means only the 15m label is mature. Both still need real fills, funding PnL, stop behavior, and repeat-event evidence.
