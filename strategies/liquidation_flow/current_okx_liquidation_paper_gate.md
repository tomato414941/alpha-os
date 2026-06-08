# Current OKX Liquidation Paper Gate

This subtracts assumed round-trip taker fees, current spread, and a simple visible-depth impact proxy from the 15m monitor-sample continuation label. It is a sizing gate, not a trade instruction.

| asset | action | size USD | gross bps | cost bps | net bps | near depth 5bps | depth usage | gate | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| XAU | short_liquidation_squeeze_watch | 100 | 1.61 | 10.23 | -8.62 | 1062653 | 0.0001 | blocked_by_cost | fee, spread, and impact proxy consume the short-window edge |
| XAU | short_liquidation_squeeze_watch | 250 | 1.61 | 10.23 | -8.62 | 1062653 | 0.0002 | blocked_by_cost | fee, spread, and impact proxy consume the short-window edge |
| XAU | short_liquidation_squeeze_watch | 500 | 1.61 | 10.24 | -8.63 | 1062653 | 0.0005 | blocked_by_cost | fee, spread, and impact proxy consume the short-window edge |
| XAU | short_liquidation_squeeze_watch | 1000 | 1.61 | 10.24 | -8.63 | 1062653 | 0.0009 | blocked_by_cost | fee, spread, and impact proxy consume the short-window edge |
| XAU | short_liquidation_squeeze_watch | 2500 | 1.61 | 10.25 | -8.64 | 1062653 | 0.0024 | blocked_by_cost | fee, spread, and impact proxy consume the short-window edge |
| XAU | short_liquidation_squeeze_watch | 5000 | 1.61 | 10.28 | -8.67 | 1062653 | 0.0047 | blocked_by_cost | fee, spread, and impact proxy consume the short-window edge |
| BTC | short_liquidation_squeeze_watch | 100 | -24.86 | 10.02 | -34.88 | 545043 | 0.0002 | blocked_by_label | monitor label is not positive |
| BTC | short_liquidation_squeeze_watch | 250 | -24.86 | 10.02 | -34.88 | 545043 | 0.0005 | blocked_by_label | monitor label is not positive |
| BTC | short_liquidation_squeeze_watch | 500 | -24.86 | 10.02 | -34.89 | 545043 | 0.0009 | blocked_by_label | monitor label is not positive |
| BTC | short_liquidation_squeeze_watch | 1000 | -24.86 | 10.03 | -34.90 | 545043 | 0.0018 | blocked_by_label | monitor label is not positive |
| BTC | short_liquidation_squeeze_watch | 2500 | -24.86 | 10.06 | -34.92 | 545043 | 0.0046 | blocked_by_label | monitor label is not positive |
| BTC | short_liquidation_squeeze_watch | 5000 | -24.86 | 10.11 | -34.97 | 545043 | 0.0092 | blocked_by_label | monitor label is not positive |
| ZEC | short_liquidation_squeeze_watch | 100 | -80.25 | 10.30 | -90.55 | 13490 | 0.0074 | blocked_by_label | monitor label is not positive |
| ZEC | short_liquidation_squeeze_watch | 250 | -80.25 | 10.41 | -90.67 | 13490 | 0.0185 | blocked_by_label | monitor label is not positive |
| ZEC | short_liquidation_squeeze_watch | 500 | -80.25 | 10.60 | -90.85 | 13490 | 0.0371 | blocked_by_label | monitor label is not positive |
| ZEC | short_liquidation_squeeze_watch | 1000 | -80.25 | 10.97 | -91.22 | 13490 | 0.0741 | blocked_by_label | monitor label is not positive |
| ZEC | short_liquidation_squeeze_watch | 2500 | -80.25 | 12.08 | -92.33 | 13490 | 0.1853 | blocked_by_label | monitor label is not positive |
| ZEC | short_liquidation_squeeze_watch | 5000 | -80.25 | 13.93 | -94.19 | 13490 | 0.3706 | blocked_by_label | monitor label is not positive |
| ETH | short_liquidation_squeeze_watch | 100 |  | 10.06 |  | 1196107 | 0.0001 | wait_for_label | no positive-direction monitor label yet |
| ETH | short_liquidation_squeeze_watch | 250 |  | 10.06 |  | 1196107 | 0.0002 | wait_for_label | no positive-direction monitor label yet |

## Interpretation

`small_paper_probe` means the current short-window label survives this rough fee/spread/depth check at the listed notional. This still omits real account fees, order-type choice, live spread changes, queue position, and stop logic.
