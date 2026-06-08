# Current OKX Liquidation Paper Gate

This subtracts assumed round-trip taker fees, current spread, and a simple visible-depth impact proxy from the 15m monitor-sample continuation label. It is a sizing gate, not a trade instruction.

| asset | action | size USD | gross bps | cost bps | net bps | near depth 5bps | depth usage | gate | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| XAU | short_liquidation_squeeze_watch | 100 | 1.61 | 10.23 | -8.62 | 1213655 | 0.0001 | blocked_by_cost | fee, spread, and impact proxy consume the short-window edge |
| XAU | short_liquidation_squeeze_watch | 250 | 1.61 | 10.23 | -8.62 | 1213655 | 0.0002 | blocked_by_cost | fee, spread, and impact proxy consume the short-window edge |
| XAU | short_liquidation_squeeze_watch | 500 | 1.61 | 10.23 | -8.63 | 1213655 | 0.0004 | blocked_by_cost | fee, spread, and impact proxy consume the short-window edge |
| XAU | short_liquidation_squeeze_watch | 1000 | 1.61 | 10.24 | -8.63 | 1213655 | 0.0008 | blocked_by_cost | fee, spread, and impact proxy consume the short-window edge |
| XAU | short_liquidation_squeeze_watch | 2500 | 1.61 | 10.25 | -8.64 | 1213655 | 0.0021 | blocked_by_cost | fee, spread, and impact proxy consume the short-window edge |
| XAU | short_liquidation_squeeze_watch | 5000 | 1.61 | 10.27 | -8.66 | 1213655 | 0.0041 | blocked_by_cost | fee, spread, and impact proxy consume the short-window edge |
| BTC | short_liquidation_squeeze_watch | 100 | -24.86 | 10.02 | -34.88 | 409894 | 0.0002 | blocked_by_label | monitor label is not positive |
| BTC | short_liquidation_squeeze_watch | 250 | -24.86 | 10.02 | -34.88 | 409894 | 0.0006 | blocked_by_label | monitor label is not positive |
| BTC | short_liquidation_squeeze_watch | 500 | -24.86 | 10.03 | -34.89 | 409894 | 0.0012 | blocked_by_label | monitor label is not positive |
| BTC | short_liquidation_squeeze_watch | 1000 | -24.86 | 10.04 | -34.90 | 409894 | 0.0024 | blocked_by_label | monitor label is not positive |
| BTC | short_liquidation_squeeze_watch | 2500 | -24.86 | 10.08 | -34.94 | 409894 | 0.0061 | blocked_by_label | monitor label is not positive |
| BTC | short_liquidation_squeeze_watch | 5000 | -24.86 | 10.14 | -35.00 | 409894 | 0.0122 | blocked_by_label | monitor label is not positive |
| ZEC | short_liquidation_squeeze_watch | 100 | -80.25 | 10.26 | -90.52 | 25589 | 0.0039 | blocked_by_label | monitor label is not positive |
| ZEC | short_liquidation_squeeze_watch | 250 | -80.25 | 10.32 | -90.58 | 25589 | 0.0098 | blocked_by_label | monitor label is not positive |
| ZEC | short_liquidation_squeeze_watch | 500 | -80.25 | 10.42 | -90.68 | 25589 | 0.0195 | blocked_by_label | monitor label is not positive |
| ZEC | short_liquidation_squeeze_watch | 1000 | -80.25 | 10.62 | -90.87 | 25589 | 0.0391 | blocked_by_label | monitor label is not positive |
| ZEC | short_liquidation_squeeze_watch | 2500 | -80.25 | 11.20 | -91.46 | 25589 | 0.0977 | blocked_by_label | monitor label is not positive |
| ZEC | short_liquidation_squeeze_watch | 5000 | -80.25 | 12.18 | -92.43 | 25589 | 0.1954 | blocked_by_label | monitor label is not positive |
| ETH | short_liquidation_squeeze_watch | 100 |  | 10.06 |  | 1019968 | 0.0001 | wait_for_label | no positive-direction monitor label yet |
| ETH | short_liquidation_squeeze_watch | 250 |  | 10.06 |  | 1019968 | 0.0002 | wait_for_label | no positive-direction monitor label yet |

## Interpretation

`small_paper_probe` means the current short-window label survives this rough fee/spread/depth check at the listed notional. This still omits real account fees, order-type choice, live spread changes, queue position, and stop logic.
