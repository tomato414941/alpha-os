# Current OKX Liquidation Intensity Paper Gate

This applies a rough OKX spread, taker-fee, and visible-depth haircut to liquidation-intensity forward labels. It is not a trade instruction.

| asset | action | label | side | size USD | label bps | cost bps | net bps | depth 10bps USD | usage | gate | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| MRVL | short_liquidation_squeeze_watch | reversal_15m_supported_pending_1h | short | 100 | 119.01 | 13.35 | 105.66 | 99968 | 0.0010 | small_paper_probe_pending_1h | wait for MRVL 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| MRVL | short_liquidation_squeeze_watch | reversal_15m_supported_pending_1h | short | 250 | 119.01 | 13.36 | 105.64 | 99968 | 0.0025 | small_paper_probe_pending_1h | wait for MRVL 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| MRVL | short_liquidation_squeeze_watch | reversal_15m_supported_pending_1h | short | 500 | 119.01 | 13.39 | 105.62 | 99968 | 0.0050 | small_paper_probe_pending_1h | wait for MRVL 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| MRVL | short_liquidation_squeeze_watch | reversal_15m_supported_pending_1h | short | 1000 | 119.01 | 13.44 | 105.57 | 99968 | 0.0100 | small_paper_probe_pending_1h | wait for MRVL 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| BEAT | short_liquidation_squeeze_watch | continuation_15m_supported_pending_1h | long | 100 | 78.13 | 10.32 | 67.80 | 10662 | 0.0094 | small_paper_probe_pending_1h | wait for BEAT 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| BEAT | short_liquidation_squeeze_watch | continuation_15m_supported_pending_1h | long | 250 | 78.13 | 10.46 | 67.66 | 10662 | 0.0234 | small_paper_probe_pending_1h | wait for BEAT 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| BEAT | short_liquidation_squeeze_watch | continuation_15m_supported_pending_1h | long | 500 | 78.13 | 10.70 | 67.43 | 10662 | 0.0469 | small_paper_probe_pending_1h | wait for BEAT 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| BEAT | short_liquidation_squeeze_watch | continuation_15m_supported_pending_1h | long | 1000 | 78.13 | 11.17 | 66.96 | 10662 | 0.0938 | small_paper_probe_pending_1h | wait for BEAT 1h label, then paper-check with fill, funding, stop, and repeat-event logs |

## Interpretation

`small_paper_probe` means a label with 1h support still survives this rough gate. `small_paper_probe_pending_1h` means only the 15m label is mature. Both still need real fills, funding PnL, stop behavior, and repeat-event evidence.
