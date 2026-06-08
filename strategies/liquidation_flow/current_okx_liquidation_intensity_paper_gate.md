# Current OKX Liquidation Intensity Paper Gate

This applies a rough OKX spread, taker-fee, and visible-depth haircut to liquidation-intensity forward labels. It is not a trade instruction.

| asset | action | label | side | size USD | label bps | cost bps | net bps | depth 10bps USD | usage | gate | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| TON | short_liquidation_squeeze_watch | reversal_15m_supported_pending_1h | short | 100 | 69.01 | 15.81 | 53.19 | 43580 | 0.0023 | small_paper_probe_pending_1h | wait for TON 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| TON | short_liquidation_squeeze_watch | reversal_15m_supported_pending_1h | short | 250 | 69.01 | 15.85 | 53.16 | 43580 | 0.0057 | small_paper_probe_pending_1h | wait for TON 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| TON | short_liquidation_squeeze_watch | reversal_15m_supported_pending_1h | short | 500 | 69.01 | 15.90 | 53.10 | 43580 | 0.0115 | small_paper_probe_pending_1h | wait for TON 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| TON | short_liquidation_squeeze_watch | reversal_15m_supported_pending_1h | short | 1000 | 69.01 | 16.02 | 52.99 | 43580 | 0.0229 | small_paper_probe_pending_1h | wait for TON 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| ONDO | long_liquidation_cascade_watch | continuation_15m_supported_pending_1h | short | 100 | 22.26 | 12.77 | 9.49 | 45223 | 0.0022 | small_paper_probe_pending_1h | wait for ONDO 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| ONDO | long_liquidation_cascade_watch | continuation_15m_supported_pending_1h | short | 250 | 22.26 | 12.80 | 9.46 | 45223 | 0.0055 | small_paper_probe_pending_1h | wait for ONDO 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| ONDO | long_liquidation_cascade_watch | continuation_15m_supported_pending_1h | short | 500 | 22.26 | 12.86 | 9.40 | 45223 | 0.0111 | small_paper_probe_pending_1h | wait for ONDO 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| ONDO | long_liquidation_cascade_watch | continuation_15m_supported_pending_1h | short | 1000 | 22.26 | 12.97 | 9.29 | 45223 | 0.0221 | small_paper_probe_pending_1h | wait for ONDO 1h label, then paper-check with fill, funding, stop, and repeat-event logs |

## Interpretation

`small_paper_probe` means a label with 1h support still survives this rough gate. `small_paper_probe_pending_1h` means only the 15m label is mature. Both still need real fills, funding PnL, stop behavior, and repeat-event evidence.
