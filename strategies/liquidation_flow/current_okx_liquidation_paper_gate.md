# Current OKX Liquidation Paper Gate

This subtracts assumed round-trip taker fees, current spread, and a simple visible-depth impact proxy from the 15m monitor-sample continuation label. It is a sizing gate, not a trade instruction.

| asset | action | size USD | gross bps | cost bps | net bps | near depth 5bps | depth usage | gate | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| XAU | short_liquidation_squeeze_watch | 100 | 1.61 | 10.23 | -8.62 | 1043633 | 0.0001 | blocked_by_cost | fee, spread, and impact proxy consume the short-window edge |
| XAU | short_liquidation_squeeze_watch | 250 | 1.61 | 10.23 | -8.62 | 1043633 | 0.0002 | blocked_by_cost | fee, spread, and impact proxy consume the short-window edge |
| XAU | short_liquidation_squeeze_watch | 500 | 1.61 | 10.23 | -8.62 | 1043633 | 0.0005 | blocked_by_cost | fee, spread, and impact proxy consume the short-window edge |
| XAU | short_liquidation_squeeze_watch | 1000 | 1.61 | 10.24 | -8.63 | 1043633 | 0.0010 | blocked_by_cost | fee, spread, and impact proxy consume the short-window edge |
| XAU | short_liquidation_squeeze_watch | 2500 | 1.61 | 10.25 | -8.64 | 1043633 | 0.0024 | blocked_by_cost | fee, spread, and impact proxy consume the short-window edge |
| XAU | short_liquidation_squeeze_watch | 5000 | 1.61 | 10.28 | -8.67 | 1043633 | 0.0048 | blocked_by_cost | fee, spread, and impact proxy consume the short-window edge |
| BTC | short_liquidation_squeeze_watch | 100 | -24.86 | 10.02 | -34.88 | 182429 | 0.0005 | blocked_by_label | monitor label is not positive |
| BTC | short_liquidation_squeeze_watch | 250 | -24.86 | 10.03 | -34.89 | 182429 | 0.0014 | blocked_by_label | monitor label is not positive |
| BTC | short_liquidation_squeeze_watch | 500 | -24.86 | 10.04 | -34.91 | 182429 | 0.0027 | blocked_by_label | monitor label is not positive |
| BTC | short_liquidation_squeeze_watch | 1000 | -24.86 | 10.07 | -34.93 | 182429 | 0.0055 | blocked_by_label | monitor label is not positive |
| BTC | short_liquidation_squeeze_watch | 2500 | -24.86 | 10.15 | -35.02 | 182429 | 0.0137 | blocked_by_label | monitor label is not positive |
| BTC | short_liquidation_squeeze_watch | 5000 | -24.86 | 10.29 | -35.15 | 182429 | 0.0274 | blocked_by_label | monitor label is not positive |
| ZEC | short_liquidation_squeeze_watch | 100 | -80.25 | 10.30 | -90.55 | 14794 | 0.0068 | blocked_by_label | monitor label is not positive |
| ZEC | short_liquidation_squeeze_watch | 250 | -80.25 | 10.40 | -90.65 | 14794 | 0.0169 | blocked_by_label | monitor label is not positive |
| ZEC | short_liquidation_squeeze_watch | 500 | -80.25 | 10.57 | -90.82 | 14794 | 0.0338 | blocked_by_label | monitor label is not positive |
| ZEC | short_liquidation_squeeze_watch | 1000 | -80.25 | 10.91 | -91.16 | 14794 | 0.0676 | blocked_by_label | monitor label is not positive |
| ZEC | short_liquidation_squeeze_watch | 2500 | -80.25 | 11.92 | -92.17 | 14794 | 0.1690 | blocked_by_label | monitor label is not positive |
| ZEC | short_liquidation_squeeze_watch | 5000 | -80.25 | 13.61 | -93.86 | 14794 | 0.3380 | blocked_by_label | monitor label is not positive |
| ETH | short_liquidation_squeeze_watch | 100 |  | 10.06 |  | 590417 | 0.0002 | wait_for_label | no positive-direction monitor label yet |
| ETH | short_liquidation_squeeze_watch | 250 |  | 10.06 |  | 590417 | 0.0004 | wait_for_label | no positive-direction monitor label yet |

## Interpretation

`small_paper_probe` means the current short-window label survives this rough fee/spread/depth check at the listed notional. This still omits real account fees, order-type choice, live spread changes, queue position, and stop logic.
