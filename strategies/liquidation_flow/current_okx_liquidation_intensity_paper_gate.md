# Current OKX Liquidation Intensity Paper Gate

This applies a rough OKX spread, taker-fee, and visible-depth haircut to liquidation-intensity forward labels. It is not a trade instruction.

| asset | action | label | side | size USD | label bps | cost bps | net bps | depth 10bps USD | usage | gate | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| LAB | short_liquidation_squeeze_watch | reversal_15m_supported_pending_1h | short | 100 | 375.60 | 11.03 | 364.57 | 3785 | 0.0264 | small_paper_probe_pending_1h | wait for LAB 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| LAB | short_liquidation_squeeze_watch | reversal_15m_supported_pending_1h | short | 250 | 375.60 | 11.43 | 364.18 | 3785 | 0.0661 | small_paper_probe_pending_1h | wait for LAB 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| LAB | short_liquidation_squeeze_watch | reversal_15m_supported_pending_1h | short | 500 | 375.60 | 12.09 | 363.51 | 3785 | 0.1321 | small_paper_probe_pending_1h | wait for LAB 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| PIPPIN | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 100 | 280.22 | 15.54 | 264.68 | 5930 | 0.0169 | small_paper_probe_pending_1h | wait for PIPPIN 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| PIPPIN | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 250 | 280.22 | 15.79 | 264.43 | 5930 | 0.0422 | small_paper_probe_pending_1h | wait for PIPPIN 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| PIPPIN | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 500 | 280.22 | 16.21 | 264.01 | 5930 | 0.0843 | small_paper_probe_pending_1h | wait for PIPPIN 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| PIPPIN | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 1000 | 280.22 | 17.06 | 263.16 | 5930 | 0.1686 | small_paper_probe_pending_1h | wait for PIPPIN 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| ALLO | short_liquidation_squeeze_watch | continuation_15m_supported_pending_1h | long | 100 | 101.41 | 10.46 | 90.95 | 4799 | 0.0208 | small_paper_probe_pending_1h | wait for ALLO 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| ALLO | short_liquidation_squeeze_watch | continuation_15m_supported_pending_1h | long | 250 | 101.41 | 10.77 | 90.64 | 4799 | 0.0521 | small_paper_probe_pending_1h | wait for ALLO 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| ALLO | short_liquidation_squeeze_watch | continuation_15m_supported_pending_1h | long | 500 | 101.41 | 11.29 | 90.12 | 4799 | 0.1042 | small_paper_probe_pending_1h | wait for ALLO 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| ALLO | short_liquidation_squeeze_watch | continuation_15m_supported_pending_1h | long | 1000 | 101.41 | 12.33 | 89.07 | 4799 | 0.2084 | small_paper_probe_pending_1h | wait for ALLO 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| BEAT | short_liquidation_squeeze_watch | reversal_15m_supported_pending_1h | short | 100 | 97.56 | 11.01 | 86.55 | 23317 | 0.0043 | small_paper_probe_pending_1h | wait for BEAT 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| BEAT | short_liquidation_squeeze_watch | reversal_15m_supported_pending_1h | short | 250 | 97.56 | 11.08 | 86.49 | 23317 | 0.0107 | small_paper_probe_pending_1h | wait for BEAT 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| BEAT | short_liquidation_squeeze_watch | reversal_15m_supported_pending_1h | short | 500 | 97.56 | 11.18 | 86.38 | 23317 | 0.0214 | small_paper_probe_pending_1h | wait for BEAT 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| BEAT | short_liquidation_squeeze_watch | reversal_15m_supported_pending_1h | short | 1000 | 97.56 | 11.40 | 86.17 | 23317 | 0.0429 | small_paper_probe_pending_1h | wait for BEAT 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| MU | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 100 | 63.15 | 10.58 | 52.56 | 97839 | 0.0010 | small_paper_probe_pending_1h | wait for MU 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| MU | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 250 | 63.15 | 10.60 | 52.55 | 97839 | 0.0026 | small_paper_probe_pending_1h | wait for MU 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| MU | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 500 | 63.15 | 10.62 | 52.52 | 97839 | 0.0051 | small_paper_probe_pending_1h | wait for MU 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| MU | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 1000 | 63.15 | 10.67 | 52.47 | 97839 | 0.0102 | small_paper_probe_pending_1h | wait for MU 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| SUI | long_liquidation_cascade_watch | reversal_15m_1h_supported | long | 100 | 63.10 | 11.35 | 51.75 | 89568 | 0.0011 | small_paper_probe | paper-check SUI liquidation intensity with fill, funding, stop, and repeat-event logs |
| SUI | long_liquidation_cascade_watch | reversal_15m_1h_supported | long | 250 | 63.10 | 11.37 | 51.73 | 89568 | 0.0028 | small_paper_probe | paper-check SUI liquidation intensity with fill, funding, stop, and repeat-event logs |
| SUI | long_liquidation_cascade_watch | reversal_15m_1h_supported | long | 500 | 63.10 | 11.40 | 51.70 | 89568 | 0.0056 | small_paper_probe | paper-check SUI liquidation intensity with fill, funding, stop, and repeat-event logs |
| SUI | long_liquidation_cascade_watch | reversal_15m_1h_supported | long | 1000 | 63.10 | 11.46 | 51.64 | 89568 | 0.0112 | small_paper_probe | paper-check SUI liquidation intensity with fill, funding, stop, and repeat-event logs |
| XAG | long_liquidation_cascade_watch | continuation_15m_supported_pending_1h | short | 100 | 51.95 | 11.49 | 40.46 | 555051 | 0.0002 | small_paper_probe_pending_1h | wait for XAG 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| XAG | long_liquidation_cascade_watch | continuation_15m_supported_pending_1h | short | 250 | 51.95 | 11.50 | 40.45 | 555051 | 0.0005 | small_paper_probe_pending_1h | wait for XAG 1h label, then paper-check with fill, funding, stop, and repeat-event logs |

## Interpretation

`small_paper_probe` means a label with 1h support still survives this rough gate. `small_paper_probe_pending_1h` means only the 15m label is mature. Both still need real fills, funding PnL, stop behavior, and repeat-event evidence.
