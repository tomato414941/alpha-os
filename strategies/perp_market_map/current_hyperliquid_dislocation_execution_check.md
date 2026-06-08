# Current Hyperliquid Dislocation Execution Check

This applies a current public-book gate to 15m-supported Hyperliquid dislocation labels. It is still not a fill model.

- rows: `60`
- paper execution probes: `22`

| asset | status | side | size | gate | gross15 | net15 | out1h | net1h | spread | depth10 | usage | reason |
| --- | --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| MOVE | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 283.67 | 267.84 | paper_1h_win | 271.61 | 4.99 | 3003 | 0.0832 | public book does not obviously block a small paper probe |
| BIO | paper_crowded_momentum_continuation_candidate | long_perp | 250 | paper_execution_probe | 280.74 | 264.81 | paper_1h_win | 94.82 | 5.60 | 7649 | 0.0327 | public book does not obviously block a small paper probe |
| BIO | paper_crowded_momentum_continuation_candidate | long_perp | 1000 | paper_execution_probe | 280.74 | 263.83 | paper_1h_win | 93.84 | 5.60 | 7649 | 0.1307 | public book does not obviously block a small paper probe |
| CC | paper_crowded_momentum_continuation_candidate | short_perp | 250 | paper_execution_probe | 101.23 | 79.26 | paper_1h_win | 65.77 | 10.19 | 1402 | 0.1783 | public book does not obviously block a small paper probe |
| CC | paper_mark_oracle_reversion_candidate | short_perp | 250 | paper_execution_probe | 101.23 | 79.26 | paper_1h_win | 65.77 | 10.19 | 1402 | 0.1783 | public book does not obviously block a small paper probe |
| CC | paper_extreme_funding_carry_candidate | short_perp | 250 | paper_execution_probe | 101.23 | 79.26 | paper_1h_win | 65.77 | 10.19 | 1402 | 0.1783 | public book does not obviously block a small paper probe |
| 2Z | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 88.50 | 74.04 | paper_1h_win | 88.45 | 3.92 | 4651 | 0.0538 | public book does not obviously block a small paper probe |
| 2Z | paper_crowded_momentum_reversal_candidate | short_perp | 1000 | paper_execution_probe | 88.50 | 72.43 | paper_1h_win | 86.84 | 3.92 | 4651 | 0.2150 | public book does not obviously block a small paper probe |
| NIL | paper_mark_oracle_reversion_candidate | long_perp | 250 | paper_execution_probe | 72.45 | 51.51 | paper_1h_win | 43.84 | 9.86 | 2311 | 0.1082 | public book does not obviously block a small paper probe |
| MORPHO | paper_crowded_momentum_continuation_candidate | long_perp | 250 | paper_execution_probe | 63.43 | 50.37 | paper_1h_win | 219.85 | 2.60 | 5522 | 0.0453 | public book does not obviously block a small paper probe |
| MORPHO | paper_crowded_momentum_continuation_candidate | long_perp | 1000 | paper_execution_probe | 63.43 | 49.01 | paper_1h_win | 218.49 | 2.60 | 5522 | 0.1811 | public book does not obviously block a small paper probe |
| APE | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 39.41 | 22.58 | paper_1h_win | 18.75 | 6.21 | 4031 | 0.0620 | public book does not obviously block a small paper probe |
| APE | paper_crowded_momentum_reversal_candidate | short_perp | 1000 | paper_execution_probe | 39.41 | 20.72 | paper_1h_win | 16.89 | 6.21 | 4031 | 0.2481 | public book does not obviously block a small paper probe |
| PROVE | paper_crowded_momentum_continuation_candidate | long_perp | 250 | paper_execution_probe | 31.25 | 16.96 | paper_1h_win | 73.33 | 3.71 | 4311 | 0.0580 | public book does not obviously block a small paper probe |
| PROVE | paper_extreme_funding_carry_candidate | long_perp | 250 | paper_execution_probe | 31.25 | 16.96 | paper_1h_win | 73.33 | 3.71 | 4311 | 0.0580 | public book does not obviously block a small paper probe |
| PROVE | paper_mark_oracle_reversion_candidate | long_perp | 250 | paper_execution_probe | 31.25 | 16.96 | paper_1h_win | 73.33 | 3.71 | 4311 | 0.0580 | public book does not obviously block a small paper probe |
| PROVE | paper_crowded_momentum_continuation_candidate | long_perp | 1000 | paper_execution_probe | 31.25 | 15.22 | paper_1h_win | 71.60 | 3.71 | 4311 | 0.2320 | public book does not obviously block a small paper probe |
| PROVE | paper_extreme_funding_carry_candidate | long_perp | 1000 | paper_execution_probe | 31.25 | 15.22 | paper_1h_win | 71.60 | 3.71 | 4311 | 0.2320 | public book does not obviously block a small paper probe |
| PROVE | paper_mark_oracle_reversion_candidate | long_perp | 1000 | paper_execution_probe | 31.25 | 15.22 | paper_1h_win | 71.60 | 3.71 | 4311 | 0.2320 | public book does not obviously block a small paper probe |
| HYPE | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 15.78 | 5.57 | paper_1h_win | 13.69 | 0.15 | 40243 | 0.0062 | public book does not obviously block a small paper probe |

## Interpretation

`paper_execution_probe` means the 15m label's gross edge still survives rough current taker fees, spread, and visible-depth impact. It still excludes queue position, partial fills, stop behavior, and repeated adverse selection.
