# OKX-Hyperliquid Event Window Score

This scores candidates by current funding event counts, not by a smooth hourly spread approximation. It is not a trade instruction.

| asset | action | scenario | long | short | OKX 8h | HL 8h | OKX 24h | HL 24h | gross 8h | gross 24h | cost | net 8h | net 24h | capacity |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BABY | thin_or_unstable_watch | very_low_fee | HlPerp | OkxSwap | 2 | 8 | 6 | 24 | 0.00098047 | 0.00294141 | 0.00266999 | -0.00168952 | 0.00027142 | 18182.63949194 |
| BTC | paper_8h_candidate | very_low_fee | OkxSwap | HlPerp | 1 | 8 | 3 | 24 | 0.00011053 | 0.00033159 | 0.00009777 | 0.00001276 | 0.00023382 | 422448.80855333 |
| BABY | thin_or_unstable_watch | low_fee | HlPerp | OkxSwap | 2 | 8 | 6 | 24 | 0.00098047 | 0.00294141 | 0.00278999 | -0.00180952 | 0.00015142 | 18182.63949194 |
| ZEC | fee_dependent_24h_monitor | very_low_fee | OkxSwap | HlPerp | 1 | 8 | 3 | 24 | 0.00027519 | 0.00082556 | 0.00070886 | -0.00043367 | 0.0001167 | 106210.05564167 |
| BTC | paper_8h_candidate | low_fee | OkxSwap | HlPerp | 1 | 8 | 3 | 24 | 0.00011053 | 0.00033159 | 0.00021777 | -0.00010724 | 0.00011382 | 422448.80855333 |
| ZEC | fee_dependent_24h_monitor | low_fee | OkxSwap | HlPerp | 1 | 8 | 3 | 24 | 0.00027519 | 0.00082556 | 0.00082886 | -0.00055367 | -0.0000033 | 106210.05564167 |
| BABY | thin_or_unstable_watch | one_bps_each | HlPerp | OkxSwap | 2 | 8 | 6 | 24 | 0.00098047 | 0.00294141 | 0.00298999 | -0.00200952 | -0.00004858 | 18182.63949194 |
| BTC | paper_8h_candidate | one_bps_each | OkxSwap | HlPerp | 1 | 8 | 3 | 24 | 0.00011053 | 0.00033159 | 0.00041777 | -0.00030724 | -0.00008618 | 422448.80855333 |
| ZEC | fee_dependent_24h_monitor | one_bps_each | OkxSwap | HlPerp | 1 | 8 | 3 | 24 | 0.00027519 | 0.00082556 | 0.00102886 | -0.00075367 | -0.0002033 | 106210.05564167 |
| JTO | active_24h_monitor | very_low_fee | OkxSwap | HlPerp | 2 | 8 | 6 | 24 | 0.00023813 | 0.0007144 | 0.0015857 | -0.00134757 | -0.0008713 | 54543.56750198 |
| JTO | active_24h_monitor | low_fee | OkxSwap | HlPerp | 2 | 8 | 6 | 24 | 0.00023813 | 0.0007144 | 0.0017057 | -0.00146757 | -0.0009913 | 54543.56750198 |
| JTO | active_24h_monitor | one_bps_each | OkxSwap | HlPerp | 2 | 8 | 6 | 24 | 0.00023813 | 0.0007144 | 0.0019057 | -0.00166757 | -0.0011913 | 54543.56750198 |

## Interpretation

A candidate that survives the smooth 24h proxy can still fail when the actual funding events inside the window are counted. This is especially important because Hyperliquid funds hourly while OKX generally funds every eight hours.
