# OKX-Hyperliquid Event Window Score

This scores candidates by current funding event counts, not by a smooth hourly spread approximation. It is not a trade instruction.

| asset | action | scenario | long | short | OKX 8h | HL 8h | OKX 24h | HL 24h | gross 8h | gross 24h | cost | net 8h | net 24h | capacity |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | paper_8h_candidate | very_low_fee | OkxSwap | HlPerp | 1 | 8 | 3 | 24 | 0.00010582 | 0.00031747 | 0.00009777 | 0.00000805 | 0.0002197 | 422448.80855333 |
| BTC | paper_8h_candidate | low_fee | OkxSwap | HlPerp | 1 | 8 | 3 | 24 | 0.00010582 | 0.00031747 | 0.00021777 | -0.00011195 | 0.0000997 | 422448.80855333 |
| ZEC | fee_dependent_24h_monitor | very_low_fee | OkxSwap | HlPerp | 1 | 8 | 3 | 24 | 0.00024083 | 0.00072249 | 0.00070886 | -0.00046803 | 0.00001363 | 106210.05564167 |
| BTC | paper_8h_candidate | one_bps_each | OkxSwap | HlPerp | 1 | 8 | 3 | 24 | 0.00010582 | 0.00031747 | 0.00041777 | -0.00031195 | -0.0001003 | 422448.80855333 |
| ZEC | fee_dependent_24h_monitor | low_fee | OkxSwap | HlPerp | 1 | 8 | 3 | 24 | 0.00024083 | 0.00072249 | 0.00082886 | -0.00058803 | -0.00010637 | 106210.05564167 |
| ZEC | fee_dependent_24h_monitor | one_bps_each | OkxSwap | HlPerp | 1 | 8 | 3 | 24 | 0.00024083 | 0.00072249 | 0.00102886 | -0.00078803 | -0.00030637 | 106210.05564167 |
| JTO | active_24h_monitor | very_low_fee | OkxSwap | HlPerp | 2 | 8 | 6 | 24 | 0.00005877 | 0.00017631 | 0.0015857 | -0.00152693 | -0.00140939 | 54543.56750198 |
| JTO | active_24h_monitor | low_fee | OkxSwap | HlPerp | 2 | 8 | 6 | 24 | 0.00005877 | 0.00017631 | 0.0017057 | -0.00164693 | -0.00152939 | 54543.56750198 |
| BABY | thin_or_unstable_watch | very_low_fee | HlPerp | OkxSwap | 2 | 8 | 6 | 24 | 0.00036862 | 0.00110587 | 0.00266999 | -0.00230137 | -0.00156412 | 18182.63949194 |
| BABY | thin_or_unstable_watch | low_fee | HlPerp | OkxSwap | 2 | 8 | 6 | 24 | 0.00036862 | 0.00110587 | 0.00278999 | -0.00242137 | -0.00168412 | 18182.63949194 |
| JTO | active_24h_monitor | one_bps_each | OkxSwap | HlPerp | 2 | 8 | 6 | 24 | 0.00005877 | 0.00017631 | 0.0019057 | -0.00184693 | -0.00172939 | 54543.56750198 |
| BABY | thin_or_unstable_watch | one_bps_each | HlPerp | OkxSwap | 2 | 8 | 6 | 24 | 0.00036862 | 0.00110587 | 0.00298999 | -0.00262137 | -0.00188412 | 18182.63949194 |

## Interpretation

A candidate that survives the smooth 24h proxy can still fail when the actual funding events inside the window are counted. This is especially important because Hyperliquid funds hourly while OKX generally funds every eight hours.
