# Current Protocol Fee Repeat Risk Check

This checks protocol-fee repeat candidates against rough spread, taker fee, funding, and visible depth. It is not a live order list.

| token | action | net 4h bps | mean 4h bps | cost bps | funding bps | depth | usage | labels | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| MORPHO/Morpho Blue | refresh_before_repeat | 97.43 | 107.99 | 10.56 | -0.00 | 6408 | 0.1561 | 1/4 | label support exists but actionability has not promoted this to repeat yet |
| CRV/Curve DEX | cost_adjusted_repeat_probe | 19.79 | 33.82 | 14.04 | -0.00 | 38721 | 0.0258 | 2/4 | repeat label survives rough spread, taker-fee, funding, and visible-depth checks |
