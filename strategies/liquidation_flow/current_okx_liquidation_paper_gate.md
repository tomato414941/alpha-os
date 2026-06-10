# Current OKX Liquidation Paper Gate

This subtracts assumed round-trip taker fees, current spread, and a simple visible-depth impact proxy from the 15m monitor-sample continuation label. It is a sizing gate, not a trade instruction.

| asset | action | size USD | gross bps | cost bps | net bps | near depth 5bps | depth usage | gate | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| XAU | short_liquidation_squeeze_watch | 100 | 1.61 | 10.23 | -8.62 | 998335 | 0.0001 | blocked_by_cost | fee, spread, and impact proxy consume the short-window edge |
| XAU | short_liquidation_squeeze_watch | 250 | 1.61 | 10.23 | -8.62 | 998335 | 0.0003 | blocked_by_cost | fee, spread, and impact proxy consume the short-window edge |
| XAU | short_liquidation_squeeze_watch | 500 | 1.61 | 10.24 | -8.63 | 998335 | 0.0005 | blocked_by_cost | fee, spread, and impact proxy consume the short-window edge |
| XAU | short_liquidation_squeeze_watch | 1000 | 1.61 | 10.24 | -8.63 | 998335 | 0.0010 | blocked_by_cost | fee, spread, and impact proxy consume the short-window edge |
| XAU | short_liquidation_squeeze_watch | 2500 | 1.61 | 10.26 | -8.65 | 998335 | 0.0025 | blocked_by_cost | fee, spread, and impact proxy consume the short-window edge |
| XAU | short_liquidation_squeeze_watch | 5000 | 1.61 | 10.28 | -8.67 | 998335 | 0.0050 | blocked_by_cost | fee, spread, and impact proxy consume the short-window edge |
| BTC | short_liquidation_squeeze_watch | 100 | -24.86 | 10.02 | -34.88 | 484882 | 0.0002 | blocked_by_label | monitor label is not positive |
| BTC | short_liquidation_squeeze_watch | 250 | -24.86 | 10.02 | -34.88 | 484882 | 0.0005 | blocked_by_label | monitor label is not positive |
| BTC | short_liquidation_squeeze_watch | 500 | -24.86 | 10.03 | -34.89 | 484882 | 0.0010 | blocked_by_label | monitor label is not positive |
| BTC | short_liquidation_squeeze_watch | 1000 | -24.86 | 10.04 | -34.90 | 484882 | 0.0021 | blocked_by_label | monitor label is not positive |
| BTC | short_liquidation_squeeze_watch | 2500 | -24.86 | 10.07 | -34.93 | 484882 | 0.0052 | blocked_by_label | monitor label is not positive |
| BTC | short_liquidation_squeeze_watch | 5000 | -24.86 | 10.12 | -34.98 | 484882 | 0.0103 | blocked_by_label | monitor label is not positive |
| ZEC | short_liquidation_squeeze_watch | 100 | -80.25 | 10.26 | -90.52 | 25685 | 0.0039 | blocked_by_label | monitor label is not positive |
| ZEC | short_liquidation_squeeze_watch | 250 | -80.25 | 10.32 | -90.57 | 25685 | 0.0097 | blocked_by_label | monitor label is not positive |
| ZEC | short_liquidation_squeeze_watch | 500 | -80.25 | 10.42 | -90.67 | 25685 | 0.0195 | blocked_by_label | monitor label is not positive |
| ZEC | short_liquidation_squeeze_watch | 1000 | -80.25 | 10.61 | -90.87 | 25685 | 0.0389 | blocked_by_label | monitor label is not positive |
| ZEC | short_liquidation_squeeze_watch | 2500 | -80.25 | 11.20 | -91.45 | 25685 | 0.0973 | blocked_by_label | monitor label is not positive |
| ZEC | short_liquidation_squeeze_watch | 5000 | -80.25 | 12.17 | -92.42 | 25685 | 0.1947 | blocked_by_label | monitor label is not positive |
| ETH | short_liquidation_squeeze_watch | 100 |  | 10.06 |  | 1201896 | 0.0001 | wait_for_label | no positive-direction monitor label yet |
| ETH | short_liquidation_squeeze_watch | 250 |  | 10.06 |  | 1201896 | 0.0002 | wait_for_label | no positive-direction monitor label yet |

## Interpretation

`small_paper_probe` means the current short-window label survives this rough fee/spread/depth check at the listed notional. This still omits real account fees, order-type choice, live spread changes, queue position, and stop logic.
