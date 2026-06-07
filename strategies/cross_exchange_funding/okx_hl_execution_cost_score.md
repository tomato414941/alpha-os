# OKX-Hyperliquid Execution Cost Score

This scores focused candidates by subtracting observed top-book taker slippage and simple fee assumptions from gross funding edge.

| asset | scenario | long | short | gross 8h | gross 24h | entry slippage bps | all-in cost | net 8h | net 24h | capacity | filled |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BABY | very_low_fee | HlPerp | OkxSwap | 0.00131017 | 0.00393051 | 12.94997144 | 0.00266999 | -0.00135982 | 0.00126052 | 18182.63949194 | True |
| BABY | low_fee | HlPerp | OkxSwap | 0.00131017 | 0.00393051 | 12.94997144 | 0.00278999 | -0.00147982 | 0.00114052 | 18182.63949194 | True |
| BABY | one_bps_each | HlPerp | OkxSwap | 0.00131017 | 0.00393051 | 12.94997144 | 0.00298999 | -0.00167982 | 0.00094052 | 18182.63949194 | True |
| JTO | very_low_fee | OkxSwap | HlPerp | 0.00069315 | 0.00207945 | 7.52849755 | 0.0015857 | -0.00089255 | 0.00049376 | 54543.56750198 | True |
| JTO | low_fee | OkxSwap | HlPerp | 0.00069315 | 0.00207945 | 7.52849755 | 0.0017057 | -0.00101255 | 0.00037376 | 54543.56750198 | True |
| ZEC | very_low_fee | OkxSwap | HlPerp | 0.00032955 | 0.00098864 | 3.14430447 | 0.00070886 | -0.00037931 | 0.00027978 | 106210.05564167 | True |
| BTC | very_low_fee | OkxSwap | HlPerp | 0.00011324 | 0.00033973 | 0.08887218 | 0.00009777 | 0.00001547 | 0.00024195 | 422448.80855333 | True |
| JTO | one_bps_each | OkxSwap | HlPerp | 0.00069315 | 0.00207945 | 7.52849755 | 0.0019057 | -0.00121255 | 0.00017376 | 54543.56750198 | True |
| ZEC | low_fee | OkxSwap | HlPerp | 0.00032955 | 0.00098864 | 3.14430447 | 0.00082886 | -0.00049931 | 0.00015978 | 106210.05564167 | True |
| BTC | low_fee | OkxSwap | HlPerp | 0.00011324 | 0.00033973 | 0.08887218 | 0.00021777 | -0.00010453 | 0.00012195 | 422448.80855333 | True |
| ZEC | one_bps_each | OkxSwap | HlPerp | 0.00032955 | 0.00098864 | 3.14430447 | 0.00102886 | -0.00069931 | -0.00004022 | 106210.05564167 | True |
| BTC | one_bps_each | OkxSwap | HlPerp | 0.00011324 | 0.00033973 | 0.08887218 | 0.00041777 | -0.00030453 | -0.00007805 | 422448.80855333 | True |

## Interpretation

A positive 24h score here is still not enough to trade. It only means the current public top book and fee assumption do not immediately erase the funding edge. Real account fee tier, maker behavior, funding-event timing, and persistence still decide whether the candidate is executable.
