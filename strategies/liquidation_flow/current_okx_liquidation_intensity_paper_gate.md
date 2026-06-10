# Current OKX Liquidation Intensity Paper Gate

This applies a rough OKX spread, taker-fee, and visible-depth haircut to liquidation-intensity forward labels. It is not a trade instruction.

| asset | action | label | side | size USD | label bps | cost bps | net bps | depth 10bps USD | usage | gate | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ALLO | short_liquidation_squeeze_watch | continuation_15m_supported_pending_1h | long | 100 | 398.72 | 10.36 | 388.36 | 8556 | 0.0117 | small_paper_probe_pending_1h | wait for ALLO 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| ALLO | short_liquidation_squeeze_watch | continuation_15m_supported_pending_1h | long | 250 | 398.72 | 10.53 | 388.19 | 8556 | 0.0292 | small_paper_probe_pending_1h | wait for ALLO 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| ALLO | short_liquidation_squeeze_watch | continuation_15m_supported_pending_1h | long | 500 | 398.72 | 10.82 | 387.89 | 8556 | 0.0584 | small_paper_probe_pending_1h | wait for ALLO 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| ALLO | short_liquidation_squeeze_watch | continuation_15m_supported_pending_1h | long | 1000 | 398.72 | 11.41 | 387.31 | 8556 | 0.1169 | small_paper_probe_pending_1h | wait for ALLO 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| FIL | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 100 | 39.64 | 11.34 | 28.30 | 54516 | 0.0018 | small_paper_probe_pending_1h | wait for FIL 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| FIL | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 250 | 39.64 | 11.36 | 28.27 | 54516 | 0.0046 | small_paper_probe_pending_1h | wait for FIL 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| FIL | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 500 | 39.64 | 11.41 | 28.23 | 54516 | 0.0092 | small_paper_probe_pending_1h | wait for FIL 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| FIL | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 1000 | 39.64 | 11.50 | 28.13 | 54516 | 0.0183 | small_paper_probe_pending_1h | wait for FIL 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| BNB | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 100 | 37.12 | 11.68 | 25.44 | 388416 | 0.0003 | small_paper_probe_pending_1h | wait for BNB 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| BNB | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 250 | 37.12 | 11.69 | 25.43 | 388416 | 0.0006 | small_paper_probe_pending_1h | wait for BNB 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| BNB | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 500 | 37.12 | 11.69 | 25.42 | 388416 | 0.0013 | small_paper_probe_pending_1h | wait for BNB 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| BNB | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 1000 | 37.12 | 11.71 | 25.41 | 388416 | 0.0026 | small_paper_probe_pending_1h | wait for BNB 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| ADA | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 100 | 36.10 | 16.02 | 20.08 | 78663 | 0.0013 | small_paper_probe_pending_1h | wait for ADA 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| ADA | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 250 | 36.10 | 16.04 | 20.07 | 78663 | 0.0032 | small_paper_probe_pending_1h | wait for ADA 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| ADA | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 500 | 36.10 | 16.07 | 20.03 | 78663 | 0.0064 | small_paper_probe_pending_1h | wait for ADA 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| ADA | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 1000 | 36.10 | 16.13 | 19.97 | 78663 | 0.0127 | small_paper_probe_pending_1h | wait for ADA 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| PEPE | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 100 | 25.42 | 13.63 | 11.78 | 177329 | 0.0006 | small_paper_probe_pending_1h | wait for PEPE 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| PEPE | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 250 | 25.42 | 13.64 | 11.77 | 177329 | 0.0014 | small_paper_probe_pending_1h | wait for PEPE 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| PEPE | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 500 | 25.42 | 13.66 | 11.76 | 177329 | 0.0028 | small_paper_probe_pending_1h | wait for PEPE 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| PEPE | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 1000 | 25.42 | 13.69 | 11.73 | 177329 | 0.0056 | small_paper_probe_pending_1h | wait for PEPE 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| BCH | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 100 | 24.47 | 14.92 | 9.55 | 58092 | 0.0017 | small_paper_probe_pending_1h | wait for BCH 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| BCH | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 250 | 24.47 | 14.95 | 9.53 | 58092 | 0.0043 | small_paper_probe_pending_1h | wait for BCH 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| BCH | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 500 | 24.47 | 14.99 | 9.48 | 58092 | 0.0086 | small_paper_probe_pending_1h | wait for BCH 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| BCH | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 1000 | 24.47 | 15.08 | 9.40 | 58092 | 0.0172 | small_paper_probe_pending_1h | wait for BCH 1h label, then paper-check with fill, funding, stop, and repeat-event logs |

## Interpretation

`small_paper_probe` means a label with 1h support still survives this rough gate. `small_paper_probe_pending_1h` means only the 15m label is mature. Both still need real fills, funding PnL, stop behavior, and repeat-event evidence.
