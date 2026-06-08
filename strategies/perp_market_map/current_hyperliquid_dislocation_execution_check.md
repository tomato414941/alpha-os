# Current Hyperliquid Dislocation Execution Check

This applies a current public-book gate to 15m-supported Hyperliquid dislocation labels. It is still not a fill model.

- rows: `60`
- paper execution probes: `24`

| asset | status | side | size | gate | gross15 | net15 | out1h | net1h | spread | depth10 | usage | reason |
| --- | --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| MOVE | paper_extreme_funding_carry_candidate | long_perp | 250 | paper_execution_probe | 174.41 | 151.62 | paper_1h_win | 1564.01 | 10.90 | 1324 | 0.1888 | public book does not obviously block a small paper probe |
| MOVE | paper_crowded_momentum_continuation_candidate | long_perp | 250 | paper_execution_probe | 174.41 | 151.62 | paper_1h_win | 1564.01 | 10.90 | 1324 | 0.1888 | public book does not obviously block a small paper probe |
| MOVE | paper_mark_oracle_reversion_candidate | long_perp | 250 | paper_execution_probe | 174.41 | 151.62 | paper_1h_win | 1564.01 | 10.90 | 1324 | 0.1888 | public book does not obviously block a small paper probe |
| NEAR | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 113.39 | 101.44 | paper_1h_win | 157.76 | 1.88 | 40418 | 0.0062 | public book does not obviously block a small paper probe |
| NEAR | paper_crowded_momentum_reversal_candidate | short_perp | 1000 | paper_execution_probe | 113.39 | 101.26 | paper_1h_win | 157.57 | 1.88 | 40418 | 0.0247 | public book does not obviously block a small paper probe |
| NEAR | paper_crowded_momentum_reversal_candidate | short_perp | 2500 | paper_execution_probe | 113.39 | 100.89 | paper_1h_win | 157.20 | 1.88 | 40418 | 0.0619 | public book does not obviously block a small paper probe |
| WLD | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 108.59 | 93.55 | paper_1h_win | 133.91 | 4.96 | 32474 | 0.0077 | public book does not obviously block a small paper probe |
| WLD | paper_crowded_momentum_reversal_candidate | short_perp | 1000 | paper_execution_probe | 108.59 | 93.32 | paper_1h_win | 133.68 | 4.96 | 32474 | 0.0308 | public book does not obviously block a small paper probe |
| WLD | paper_crowded_momentum_reversal_candidate | short_perp | 2500 | paper_execution_probe | 108.59 | 92.86 | paper_1h_win | 133.21 | 4.96 | 32474 | 0.0770 | public book does not obviously block a small paper probe |
| BCH | paper_crowded_momentum_continuation_candidate | short_perp | 250 | paper_execution_probe | 101.60 | 89.11 | paper_1h_win | 49.54 | 2.43 | 39016 | 0.0064 | public book does not obviously block a small paper probe |
| BCH | paper_crowded_momentum_continuation_candidate | short_perp | 1000 | paper_execution_probe | 101.60 | 88.92 | paper_1h_win | 49.35 | 2.43 | 39016 | 0.0256 | public book does not obviously block a small paper probe |
| BCH | paper_crowded_momentum_continuation_candidate | short_perp | 2500 | paper_execution_probe | 101.60 | 88.53 | paper_1h_win | 48.97 | 2.43 | 39016 | 0.0641 | public book does not obviously block a small paper probe |
| LIT | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 98.77 | 84.25 | paper_1h_win | 303.96 | 3.99 | 4782 | 0.0523 | public book does not obviously block a small paper probe |
| LIT | paper_crowded_momentum_reversal_candidate | short_perp | 1000 | paper_execution_probe | 98.77 | 82.68 | paper_1h_win | 302.39 | 3.99 | 4782 | 0.2091 | public book does not obviously block a small paper probe |
| INJ | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 94.02 | 81.06 | paper_1h_win | 121.48 | 2.65 | 8123 | 0.0308 | public book does not obviously block a small paper probe |
| INJ | paper_crowded_momentum_reversal_candidate | short_perp | 1000 | paper_execution_probe | 94.02 | 80.14 | paper_1h_win | 120.56 | 2.65 | 8123 | 0.1231 | public book does not obviously block a small paper probe |
| kLUNC | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 93.21 | 75.08 | paper_1h_win | 173.20 | 6.17 | 1275 | 0.1961 | public book does not obviously block a small paper probe |
| CC | paper_crowded_momentum_reversal_candidate | long_perp | 250 | paper_execution_probe | 70.14 | 56.35 | paper_1h_win | 136.18 | 3.19 | 4166 | 0.0600 | public book does not obviously block a small paper probe |
| CC | paper_crowded_momentum_reversal_candidate | long_perp | 1000 | paper_execution_probe | 70.14 | 54.55 | paper_1h_win | 134.38 | 3.19 | 4166 | 0.2401 | public book does not obviously block a small paper probe |
| ONDO | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 65.66 | 53.67 | paper_1h_win | 52.11 | 1.94 | 54677 | 0.0046 | public book does not obviously block a small paper probe |

## Interpretation

`paper_execution_probe` means the 15m label's gross edge still survives rough current taker fees, spread, and visible-depth impact. It still excludes queue position, partial fills, stop behavior, and repeated adverse selection.
