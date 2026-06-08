# Current Hyperliquid Dislocation Execution Check

This applies a current public-book gate to 15m-supported Hyperliquid dislocation labels. It is still not a fill model.

- rows: `60`
- paper execution probes: `22`

| asset | status | side | size | gate | gross15 | net15 | out1h | net1h | spread | depth10 | usage | reason |
| --- | --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| NEAR | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 113.39 | 100.10 | paper_1h_win | 156.41 | 3.26 | 72485 | 0.0034 | public book does not obviously block a small paper probe |
| NEAR | paper_crowded_momentum_reversal_candidate | short_perp | 1000 | paper_execution_probe | 113.39 | 100.00 | paper_1h_win | 156.31 | 3.26 | 72485 | 0.0138 | public book does not obviously block a small paper probe |
| NEAR | paper_crowded_momentum_reversal_candidate | short_perp | 2500 | paper_execution_probe | 113.39 | 99.79 | paper_1h_win | 156.10 | 3.26 | 72485 | 0.0345 | public book does not obviously block a small paper probe |
| WLD | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 108.59 | 96.80 | paper_1h_win | 137.16 | 1.73 | 45982 | 0.0054 | public book does not obviously block a small paper probe |
| WLD | paper_crowded_momentum_reversal_candidate | short_perp | 1000 | paper_execution_probe | 108.59 | 96.64 | paper_1h_win | 137.00 | 1.73 | 45982 | 0.0217 | public book does not obviously block a small paper probe |
| WLD | paper_crowded_momentum_reversal_candidate | short_perp | 2500 | paper_execution_probe | 108.59 | 96.31 | paper_1h_win | 136.67 | 1.73 | 45982 | 0.0544 | public book does not obviously block a small paper probe |
| NXPC | paper_crowded_momentum_continuation_candidate | long_perp | 250 | paper_execution_probe | 110.80 | 93.25 | paper_1h_win | 27.30 | 6.33 | 2045 | 0.1223 | public book does not obviously block a small paper probe |
| BCH | paper_crowded_momentum_continuation_candidate | short_perp | 250 | paper_execution_probe | 101.60 | 91.05 | paper_1h_win | 51.48 | 0.49 | 38064 | 0.0066 | public book does not obviously block a small paper probe |
| BCH | paper_crowded_momentum_continuation_candidate | short_perp | 1000 | paper_execution_probe | 101.60 | 90.85 | paper_1h_win | 51.29 | 0.49 | 38064 | 0.0263 | public book does not obviously block a small paper probe |
| BCH | paper_crowded_momentum_continuation_candidate | short_perp | 2500 | paper_execution_probe | 101.60 | 90.46 | paper_1h_win | 50.89 | 0.49 | 38064 | 0.0657 | public book does not obviously block a small paper probe |
| LIT | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 98.77 | 80.92 | paper_1h_win | 300.63 | 7.22 | 3961 | 0.0631 | public book does not obviously block a small paper probe |
| MANTA | paper_crowded_momentum_continuation_candidate | short_perp | 250 | paper_execution_probe | 92.34 | 79.22 | paper_1h_win | 55.58 | 2.63 | 5034 | 0.0497 | public book does not obviously block a small paper probe |
| MANTA | paper_crowded_momentum_continuation_candidate | short_perp | 1000 | paper_execution_probe | 92.34 | 77.73 | paper_1h_win | 54.09 | 2.63 | 5034 | 0.1986 | public book does not obviously block a small paper probe |
| INJ | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 94.02 | 74.86 | paper_1h_win | 115.27 | 8.46 | 3558 | 0.0703 | public book does not obviously block a small paper probe |
| CC | paper_crowded_momentum_reversal_candidate | long_perp | 250 | paper_execution_probe | 70.14 | 51.64 | paper_1h_win | 131.47 | 7.64 | 2911 | 0.0859 | public book does not obviously block a small paper probe |
| KAS | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 67.11 | 51.06 | paper_1h_win | 47.81 | 5.18 | 2860 | 0.0874 | public book does not obviously block a small paper probe |
| ONDO | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 65.66 | 50.75 | paper_1h_win | 49.19 | 4.67 | 10596 | 0.0236 | public book does not obviously block a small paper probe |
| ONDO | paper_crowded_momentum_reversal_candidate | short_perp | 1000 | paper_execution_probe | 65.66 | 50.04 | paper_1h_win | 48.48 | 4.67 | 10596 | 0.0944 | public book does not obviously block a small paper probe |
| ONDO | paper_crowded_momentum_reversal_candidate | short_perp | 2500 | paper_execution_probe | 65.66 | 48.63 | paper_1h_win | 47.07 | 4.67 | 10596 | 0.2359 | public book does not obviously block a small paper probe |
| MEGA | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 60.78 | 43.96 | paper_1h_win | 44.46 | 5.79 | 2420 | 0.1033 | public book does not obviously block a small paper probe |

## Interpretation

`paper_execution_probe` means the 15m label's gross edge still survives rough current taker fees, spread, and visible-depth impact. It still excludes queue position, partial fills, stop behavior, and repeated adverse selection.
