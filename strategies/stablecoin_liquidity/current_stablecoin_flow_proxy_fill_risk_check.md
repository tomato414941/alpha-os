# Current Stablecoin Flow Proxy Fill Risk Check

This checks stablecoin chain-liquidity proxy mark outcomes against rough OKX spread, taker fee, funding, visible depth, and adverse-excursion assumptions. It is not a live fill report.

| ticket | asset | decision | notional | dir bps | cost bps | funding bps | net bps | depth 10bps | usage | MAE bps | MFE bps | stop 50 | action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| stablecoin-flow-polygon-pol | POL | paper_short | 1000 | 302.58 | 11.28 | -0.02 | 291.28 | 8601 | 0.1163 | 180.31 | 321.11 | yes | depth_too_thin_for_1k_probe | 1k notional consumes more than 10% of visible 10bps depth |
| stablecoin-flow-hyperliquid-l1-hype | HYPE | paper_long | 1000 | 12.61 | 11.58 | -0.59 | 0.43 | 135463 | 0.0074 | -105.59 | 154.45 | no | stop_risk_blocks_probe | mark edge survived cost but would have breached a rough 50bps adverse-excursion stop |
| stablecoin-flow-solana-sol | SOL | paper_long | 1000 | -50.33 | 11.50 | 3.02 | -58.81 | 973407 | 0.0010 | -106.59 | 81.42 | no | cost_adjusted_edge_failed | mark edge does not survive rough spread, taker-fee, and funding haircut |
| stablecoin-flow-arbitrum-arb | ARB | paper_long | 1000 | -104.24 | 11.22 | -3.42 | -118.88 | 12684 | 0.0788 | -150.97 | 49.13 | no | cost_adjusted_edge_failed | mark edge does not survive rough spread, taker-fee, and funding haircut |
