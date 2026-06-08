# Current OKX Liquidation Paper Gate

This subtracts assumed round-trip taker fees, current spread, and a simple visible-depth impact proxy from the 15m monitor-sample continuation label. It is a sizing gate, not a trade instruction.

| asset | action | size USD | gross bps | cost bps | net bps | near depth 5bps | depth usage | gate | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| XAU | short_liquidation_squeeze_watch | 100 | 1.61 | 10.23 | -8.62 | 1214830 | 0.0001 | blocked_by_cost | fee, spread, and impact proxy consume the short-window edge |
| XAU | short_liquidation_squeeze_watch | 250 | 1.61 | 10.23 | -8.62 | 1214830 | 0.0002 | blocked_by_cost | fee, spread, and impact proxy consume the short-window edge |
| XAU | short_liquidation_squeeze_watch | 500 | 1.61 | 10.23 | -8.63 | 1214830 | 0.0004 | blocked_by_cost | fee, spread, and impact proxy consume the short-window edge |
| XAU | short_liquidation_squeeze_watch | 1000 | 1.61 | 10.24 | -8.63 | 1214830 | 0.0008 | blocked_by_cost | fee, spread, and impact proxy consume the short-window edge |
| XAU | short_liquidation_squeeze_watch | 2500 | 1.61 | 10.25 | -8.64 | 1214830 | 0.0021 | blocked_by_cost | fee, spread, and impact proxy consume the short-window edge |
| XAU | short_liquidation_squeeze_watch | 5000 | 1.61 | 10.27 | -8.66 | 1214830 | 0.0041 | blocked_by_cost | fee, spread, and impact proxy consume the short-window edge |
| BTC | short_liquidation_squeeze_watch | 100 | -24.86 | 10.02 | -34.88 | 358952 | 0.0003 | blocked_by_label | monitor label is not positive |
| BTC | short_liquidation_squeeze_watch | 250 | -24.86 | 10.02 | -34.89 | 358952 | 0.0007 | blocked_by_label | monitor label is not positive |
| BTC | short_liquidation_squeeze_watch | 500 | -24.86 | 10.03 | -34.89 | 358952 | 0.0014 | blocked_by_label | monitor label is not positive |
| BTC | short_liquidation_squeeze_watch | 1000 | -24.86 | 10.04 | -34.91 | 358952 | 0.0028 | blocked_by_label | monitor label is not positive |
| BTC | short_liquidation_squeeze_watch | 2500 | -24.86 | 10.09 | -34.95 | 358952 | 0.0070 | blocked_by_label | monitor label is not positive |
| BTC | short_liquidation_squeeze_watch | 5000 | -24.86 | 10.16 | -35.02 | 358952 | 0.0139 | blocked_by_label | monitor label is not positive |
| ZEC | short_liquidation_squeeze_watch | 100 | -80.25 | 10.27 | -90.53 | 29648 | 0.0034 | blocked_by_label | monitor label is not positive |
| ZEC | short_liquidation_squeeze_watch | 250 | -80.25 | 10.32 | -90.58 | 29648 | 0.0084 | blocked_by_label | monitor label is not positive |
| ZEC | short_liquidation_squeeze_watch | 500 | -80.25 | 10.41 | -90.66 | 29648 | 0.0169 | blocked_by_label | monitor label is not positive |
| ZEC | short_liquidation_squeeze_watch | 1000 | -80.25 | 10.58 | -90.83 | 29648 | 0.0337 | blocked_by_label | monitor label is not positive |
| ZEC | short_liquidation_squeeze_watch | 2500 | -80.25 | 11.08 | -91.34 | 29648 | 0.0843 | blocked_by_label | monitor label is not positive |
| ZEC | short_liquidation_squeeze_watch | 5000 | -80.25 | 11.92 | -92.18 | 29648 | 0.1686 | blocked_by_label | monitor label is not positive |
| ETH | short_liquidation_squeeze_watch | 100 |  | 10.06 |  | 947889 | 0.0001 | wait_for_label | no positive-direction monitor label yet |
| ETH | short_liquidation_squeeze_watch | 250 |  | 10.06 |  | 947889 | 0.0003 | wait_for_label | no positive-direction monitor label yet |

## Interpretation

`small_paper_probe` means the current short-window label survives this rough fee/spread/depth check at the listed notional. This still omits real account fees, order-type choice, live spread changes, queue position, and stop logic.
