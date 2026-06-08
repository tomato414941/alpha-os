# Current OKX Liquidation Intensity Paper Gate

This applies a rough OKX spread, taker-fee, and visible-depth haircut to liquidation-intensity forward labels. It is not a trade instruction.

| asset | action | label | side | size USD | label bps | cost bps | net bps | depth 10bps USD | usage | gate | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| LAB | short_liquidation_squeeze_watch | reversal_15m_supported_pending_1h | short | 100 | 375.60 | 10.83 | 364.78 | 12849 | 0.0078 | small_paper_probe_pending_1h | wait for LAB 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| LAB | short_liquidation_squeeze_watch | reversal_15m_supported_pending_1h | short | 250 | 375.60 | 10.94 | 364.66 | 12849 | 0.0195 | small_paper_probe_pending_1h | wait for LAB 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| LAB | short_liquidation_squeeze_watch | reversal_15m_supported_pending_1h | short | 500 | 375.60 | 11.14 | 364.47 | 12849 | 0.0389 | small_paper_probe_pending_1h | wait for LAB 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| LAB | short_liquidation_squeeze_watch | reversal_15m_supported_pending_1h | short | 1000 | 375.60 | 11.53 | 364.08 | 12849 | 0.0778 | small_paper_probe_pending_1h | wait for LAB 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| PIPPIN | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 100 | 280.22 | 15.60 | 264.62 | 4701 | 0.0213 | small_paper_probe_pending_1h | wait for PIPPIN 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| PIPPIN | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 250 | 280.22 | 15.92 | 264.30 | 4701 | 0.0532 | small_paper_probe_pending_1h | wait for PIPPIN 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| PIPPIN | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 500 | 280.22 | 16.45 | 263.77 | 4701 | 0.1064 | small_paper_probe_pending_1h | wait for PIPPIN 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| PIPPIN | long_liquidation_cascade_watch | reversal_15m_supported_pending_1h | long | 1000 | 280.22 | 17.51 | 262.71 | 4701 | 0.2127 | small_paper_probe_pending_1h | wait for PIPPIN 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| ALLO | short_liquidation_squeeze_watch | continuation_15m_supported_pending_1h | long | 100 | 101.41 | 10.40 | 91.00 | 6206 | 0.0161 | small_paper_probe_pending_1h | wait for ALLO 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| ALLO | short_liquidation_squeeze_watch | continuation_15m_supported_pending_1h | long | 250 | 101.41 | 10.65 | 90.76 | 6206 | 0.0403 | small_paper_probe_pending_1h | wait for ALLO 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| ALLO | short_liquidation_squeeze_watch | continuation_15m_supported_pending_1h | long | 500 | 101.41 | 11.05 | 90.36 | 6206 | 0.0806 | small_paper_probe_pending_1h | wait for ALLO 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| ALLO | short_liquidation_squeeze_watch | continuation_15m_supported_pending_1h | long | 1000 | 101.41 | 11.85 | 89.55 | 6206 | 0.1611 | small_paper_probe_pending_1h | wait for ALLO 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| BEAT | short_liquidation_squeeze_watch | reversal_15m_supported_pending_1h | short | 100 | 97.56 | 10.76 | 86.80 | 18598 | 0.0054 | small_paper_probe_pending_1h | wait for BEAT 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| BEAT | short_liquidation_squeeze_watch | reversal_15m_supported_pending_1h | short | 250 | 97.56 | 10.85 | 86.72 | 18598 | 0.0134 | small_paper_probe_pending_1h | wait for BEAT 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| BEAT | short_liquidation_squeeze_watch | reversal_15m_supported_pending_1h | short | 500 | 97.56 | 10.98 | 86.58 | 18598 | 0.0269 | small_paper_probe_pending_1h | wait for BEAT 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| BEAT | short_liquidation_squeeze_watch | reversal_15m_supported_pending_1h | short | 1000 | 97.56 | 11.25 | 86.31 | 18598 | 0.0538 | small_paper_probe_pending_1h | wait for BEAT 1h label, then paper-check with fill, funding, stop, and repeat-event logs |
| MU | long_liquidation_cascade_watch | reversal_15m_1h_supported | long | 100 | 63.15 | 10.13 | 53.01 | 53818 | 0.0019 | small_paper_probe | paper-check MU liquidation intensity with fill, funding, stop, and repeat-event logs |
| MU | long_liquidation_cascade_watch | reversal_15m_1h_supported | long | 250 | 63.15 | 10.16 | 52.99 | 53818 | 0.0046 | small_paper_probe | paper-check MU liquidation intensity with fill, funding, stop, and repeat-event logs |
| MU | long_liquidation_cascade_watch | reversal_15m_1h_supported | long | 500 | 63.15 | 10.21 | 52.94 | 53818 | 0.0093 | small_paper_probe | paper-check MU liquidation intensity with fill, funding, stop, and repeat-event logs |
| MU | long_liquidation_cascade_watch | reversal_15m_1h_supported | long | 1000 | 63.15 | 10.30 | 52.85 | 53818 | 0.0186 | small_paper_probe | paper-check MU liquidation intensity with fill, funding, stop, and repeat-event logs |
| SUI | long_liquidation_cascade_watch | reversal_15m_1h_supported | long | 100 | 63.10 | 11.36 | 51.74 | 101323 | 0.0010 | small_paper_probe | paper-check SUI liquidation intensity with fill, funding, stop, and repeat-event logs |
| SUI | long_liquidation_cascade_watch | reversal_15m_1h_supported | long | 250 | 63.10 | 11.37 | 51.73 | 101323 | 0.0025 | small_paper_probe | paper-check SUI liquidation intensity with fill, funding, stop, and repeat-event logs |
| SUI | long_liquidation_cascade_watch | reversal_15m_1h_supported | long | 500 | 63.10 | 11.40 | 51.70 | 101323 | 0.0049 | small_paper_probe | paper-check SUI liquidation intensity with fill, funding, stop, and repeat-event logs |
| SUI | long_liquidation_cascade_watch | reversal_15m_1h_supported | long | 1000 | 63.10 | 11.45 | 51.65 | 101323 | 0.0099 | small_paper_probe | paper-check SUI liquidation intensity with fill, funding, stop, and repeat-event logs |
| XAG | long_liquidation_cascade_watch | continuation_15m_supported_pending_1h | short | 100 | 51.95 | 11.49 | 40.46 | 567935 | 0.0002 | small_paper_probe_pending_1h | wait for XAG 1h label, then paper-check with fill, funding, stop, and repeat-event logs |

## Interpretation

`small_paper_probe` means a label with 1h support still survives this rough gate. `small_paper_probe_pending_1h` means only the 15m label is mature. Both still need real fills, funding PnL, stop behavior, and repeat-event evidence.
