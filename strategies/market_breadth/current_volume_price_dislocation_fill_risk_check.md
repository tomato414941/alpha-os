# Current Volume Price Dislocation Fill Risk Check

This checks market-breadth mark wins against rough Hyperliquid spread, taker fee, funding, visible depth, and 1m candle path risk. It is not a live fill report.

| ticket | symbol | dir bps | cost bps | funding bps | net bps | depth 10bps | usage | MAE | MFE | stop50 | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| market-breadth-eigen-long-reversal | EIGEN | 205.44 | 15.55 | -0.03 | 189.86 | 1180 | 0.2118 | -22.21 | 260.97 | yes | depth_too_thin_for_250_probe | 250 USD notional consumes more than 10% of visible 10bps depth |
| market-breadth-hype-long-reversal | HYPE | 76.45 | 8.18 | -0.03 | 68.24 | 119078 | 0.0021 | -6.46 | 94.74 | yes | cost_adjusted_probe_survived | mark edge survives rough cost, funding, visible-depth, and 50bps adverse-excursion checks |
| market-breadth-link-long-reversal | LINK | 8.01 | 9.59 | -0.03 | -1.61 | 28289 | 0.0088 | -15.28 | 20.04 | yes | cost_adjusted_edge_failed | mark edge does not survive rough spread, taker-fee, funding, and visible-depth impact |
