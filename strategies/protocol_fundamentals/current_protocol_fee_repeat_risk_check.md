# Current Protocol Fee Repeat Risk Check

This checks protocol-fee repeat candidates against rough spread, taker fee, funding, and visible depth. It is not a live order list.

| token | action | net 4h bps | mean 4h bps | cost bps | funding bps | depth | usage | labels | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| MORPHO/Morpho Blue | refresh_before_repeat | 94.16 | 107.99 | 13.82 | -0.00 | 6508 | 0.1537 | 1/4 | label support exists but actionability has not promoted this to repeat yet |
| CRV/Curve DEX | cost_adjusted_repeat_probe | 23.32 | 33.82 | 10.50 | -0.00 | 23800 | 0.0420 | 2/4 | repeat label survives rough spread, taker-fee, funding, and visible-depth checks |
