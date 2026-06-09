# Current Protocol Fee Repeat Fill Risk Check

This checks protocol-fee repeat mark wins against rough Hyperliquid spread, taker fee, funding, visible depth, and 1m candle path risk. It is not a live fill report.

| ticket | subject | dir bps | cost bps | funding bps | net bps | depth 10bps | usage | MAE | MFE | stop50 | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| protocol-fee-repeat-crv-curve-dex | CRV/Curve DEX | 118.57 | 12.94 | -0.03 | 105.60 | 24655 | 0.0406 | -60.04 | 27.25 | no | stop_risk_blocks_repeat | mark edge survived cost but would have breached a rough 50bps adverse-excursion stop |
