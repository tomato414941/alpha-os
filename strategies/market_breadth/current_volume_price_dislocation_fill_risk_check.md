# Current Volume Price Dislocation Fill Risk Check

This checks market-breadth mark wins against rough Hyperliquid spread, taker fee, funding, visible depth, and 1m candle path risk. It is not a live fill report.

| ticket | symbol | dir bps | cost bps | funding bps | net bps | depth 10bps | usage | MAE | MFE | stop50 | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| market-breadth-eigen-long-reversal | EIGEN | 83.29 | 15.09 | -0.11 | 68.09 | 1588 | 0.1574 | -22.21 | 260.97 | yes | depth_too_thin_for_250_probe | 250 USD notional consumes more than 10% of visible 10bps depth |
