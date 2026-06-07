# Current Hyperliquid Dislocation Execution Check

This applies a current public-book gate to 15m-supported Hyperliquid dislocation labels. It is still not a fill model.

- rows: `60`
- paper execution probes: `35`

| asset | status | side | size | gate | gross15 | cost | conservative15 | spread | depth10 | usage | reason |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BRETT | paper_mark_oracle_reversion_candidate | long_perp | 250 | paper_execution_probe | 93.08 | 16.79 | 76.29 | 5.27 | 1652 | 0.1514 | public book does not obviously block a small paper probe |
| ZRO | paper_extreme_funding_carry_candidate | short_perp | 250 | paper_execution_probe | 49.48 | 11.37 | 38.11 | 1.22 | 16784 | 0.0149 | public book does not obviously block a small paper probe |
| MON | paper_crowded_momentum_continuation_candidate | long_perp | 250 | paper_execution_probe | 50.69 | 12.58 | 38.11 | 2.27 | 8096 | 0.0309 | public book does not obviously block a small paper probe |
| ZRO | paper_extreme_funding_carry_candidate | short_perp | 1000 | paper_execution_probe | 49.48 | 11.82 | 37.67 | 1.22 | 16784 | 0.0596 | public book does not obviously block a small paper probe |
| MON | paper_crowded_momentum_continuation_candidate | long_perp | 1000 | paper_execution_probe | 50.69 | 13.50 | 37.18 | 2.27 | 8096 | 0.1235 | public book does not obviously block a small paper probe |
| MEGA | paper_crowded_momentum_continuation_candidate | long_perp | 250 | paper_execution_probe | 52.91 | 16.13 | 36.78 | 5.39 | 3358 | 0.0745 | public book does not obviously block a small paper probe |
| ZRO | paper_extreme_funding_carry_candidate | short_perp | 2500 | paper_execution_probe | 49.48 | 12.71 | 36.77 | 1.22 | 16784 | 0.1489 | public book does not obviously block a small paper probe |
| WLD | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 43.58 | 17.15 | 26.43 | 7.08 | 34338 | 0.0073 | public book does not obviously block a small paper probe |
| WLD | paper_crowded_momentum_reversal_candidate | short_perp | 1000 | paper_execution_probe | 43.58 | 17.37 | 26.21 | 7.08 | 34338 | 0.0291 | public book does not obviously block a small paper probe |
| TIA | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 37.10 | 11.04 | 26.06 | 0.63 | 6060 | 0.0413 | public book does not obviously block a small paper probe |
| WLD | paper_crowded_momentum_reversal_candidate | short_perp | 2500 | paper_execution_probe | 43.58 | 17.81 | 25.77 | 7.08 | 34338 | 0.0728 | public book does not obviously block a small paper probe |
| TIA | paper_crowded_momentum_reversal_candidate | short_perp | 1000 | paper_execution_probe | 37.10 | 12.28 | 24.82 | 0.63 | 6060 | 0.1650 | public book does not obviously block a small paper probe |
| AIXBT | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 40.70 | 16.34 | 24.36 | 5.45 | 2833 | 0.0882 | public book does not obviously block a small paper probe |
| MANTA | paper_crowded_momentum_continuation_candidate | short_perp | 250 | paper_execution_probe | 39.27 | 15.91 | 23.36 | 5.27 | 3918 | 0.0638 | public book does not obviously block a small paper probe |
| DASH | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 37.66 | 15.40 | 22.26 | 5.15 | 9964 | 0.0251 | public book does not obviously block a small paper probe |
| JTO | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 40.53 | 18.75 | 21.78 | 6.50 | 1112 | 0.2248 | public book does not obviously block a small paper probe |
| DASH | paper_crowded_momentum_reversal_candidate | short_perp | 1000 | paper_execution_probe | 37.66 | 16.15 | 21.51 | 5.15 | 9964 | 0.1004 | public book does not obviously block a small paper probe |
| ONDO | paper_crowded_momentum_continuation_candidate | long_perp | 250 | paper_execution_probe | 30.56 | 10.88 | 19.68 | 0.58 | 8302 | 0.0301 | public book does not obviously block a small paper probe |
| ADA | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 31.97 | 13.05 | 18.92 | 3.03 | 106635 | 0.0023 | public book does not obviously block a small paper probe |
| ADA | paper_crowded_momentum_reversal_candidate | short_perp | 1000 | paper_execution_probe | 31.97 | 13.12 | 18.85 | 3.03 | 106635 | 0.0094 | public book does not obviously block a small paper probe |

## Interpretation

`paper_execution_probe` means the 15m label's gross edge still survives rough current taker fees, spread, and visible-depth impact. It still excludes queue position, partial fills, stop behavior, and repeated adverse selection.
