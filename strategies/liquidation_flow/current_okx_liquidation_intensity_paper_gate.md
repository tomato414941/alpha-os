# Current OKX Liquidation Intensity Paper Gate

This applies a rough OKX spread, taker-fee, and visible-depth haircut to liquidation-intensity forward labels. It is not a trade instruction.

| asset | action | label | side | size USD | label bps | cost bps | net bps | depth 10bps USD | usage | gate | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |

## Interpretation

`small_paper_probe` means a label with 1h support still survives this rough gate. `small_paper_probe_pending_1h` means only the 15m label is mature. Both still need real fills, funding PnL, stop behavior, and repeat-event evidence.
