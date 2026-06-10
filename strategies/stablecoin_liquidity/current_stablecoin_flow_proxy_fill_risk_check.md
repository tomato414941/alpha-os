# Current Stablecoin Flow Proxy Fill Risk Check

This checks stablecoin chain-liquidity proxy mark outcomes against rough OKX spread, taker fee, funding, visible depth, and adverse-excursion assumptions. It is not a live fill report.

| ticket | asset | decision | notional | dir bps | cost bps | funding bps | net bps | depth 10bps | usage | MAE bps | MFE bps | stop 50 | action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| stablecoin-flow-polygon-pol | POL | paper_short | 1000 | 561.94 | 11.28 | -0.03 | 550.63 | 8601 | 0.1163 | 180.31 | 561.94 | yes | depth_too_thin_for_1k_probe | 1k notional consumes more than 10% of visible 10bps depth |
| stablecoin-flow-solana-sol | SOL | paper_long | 1000 | -325.68 | 11.50 | 4.42 | -332.77 | 973407 | 0.0010 | -333.09 | 81.42 | no | cost_adjusted_edge_failed | mark edge does not survive rough spread, taker-fee, and funding haircut |
| stablecoin-flow-arbitrum-arb | ARB | paper_long | 1000 | -393.00 | 11.22 | -5.14 | -409.36 | 12684 | 0.0788 | -406.18 | 49.13 | no | cost_adjusted_edge_failed | mark edge does not survive rough spread, taker-fee, and funding haircut |
