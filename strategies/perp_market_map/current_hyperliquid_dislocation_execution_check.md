# Current Hyperliquid Dislocation Execution Check

This applies a current public-book gate to 15m-supported Hyperliquid dislocation labels. It is still not a fill model.

- rows: `60`
- paper execution probes: `52`

| asset | status | side | size | gate | gross15 | cost | conservative15 | spread | depth10 | usage | reason |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| DASH | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 67.22 | 17.54 | 49.69 | 7.03 | 4892 | 0.0511 | public book does not obviously block a small paper probe |
| DASH | paper_crowded_momentum_reversal_candidate | short_perp | 1000 | paper_execution_probe | 67.22 | 19.07 | 48.15 | 7.03 | 4892 | 0.2044 | public book does not obviously block a small paper probe |
| ZEC | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 58.18 | 13.47 | 44.71 | 3.43 | 67026 | 0.0037 | public book does not obviously block a small paper probe |
| ZEC | paper_crowded_momentum_reversal_candidate | short_perp | 1000 | paper_execution_probe | 58.18 | 13.58 | 44.59 | 3.43 | 67026 | 0.0149 | public book does not obviously block a small paper probe |
| ZEC | paper_crowded_momentum_reversal_candidate | short_perp | 2500 | paper_execution_probe | 58.18 | 13.81 | 44.37 | 3.43 | 67026 | 0.0373 | public book does not obviously block a small paper probe |
| SPX | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 51.87 | 11.13 | 40.74 | 0.95 | 13717 | 0.0182 | public book does not obviously block a small paper probe |
| SPX | paper_crowded_momentum_reversal_candidate | short_perp | 1000 | paper_execution_probe | 51.87 | 11.68 | 40.20 | 0.95 | 13717 | 0.0729 | public book does not obviously block a small paper probe |
| WLD | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 51.23 | 11.53 | 39.70 | 1.50 | 70573 | 0.0035 | public book does not obviously block a small paper probe |
| WLD | paper_crowded_momentum_reversal_candidate | short_perp | 1000 | paper_execution_probe | 51.23 | 11.64 | 39.59 | 1.50 | 70573 | 0.0142 | public book does not obviously block a small paper probe |
| WLD | paper_crowded_momentum_reversal_candidate | short_perp | 2500 | paper_execution_probe | 51.23 | 11.85 | 39.38 | 1.50 | 70573 | 0.0354 | public book does not obviously block a small paper probe |
| SPX | paper_crowded_momentum_reversal_candidate | short_perp | 2500 | paper_execution_probe | 51.87 | 12.77 | 39.10 | 0.95 | 13717 | 0.1823 | public book does not obviously block a small paper probe |
| SOL | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 47.76 | 10.15 | 37.61 | 0.15 | 923704 | 0.0003 | public book does not obviously block a small paper probe |
| SOL | paper_crowded_momentum_reversal_candidate | short_perp | 1000 | paper_execution_probe | 47.76 | 10.16 | 37.60 | 0.15 | 923704 | 0.0011 | public book does not obviously block a small paper probe |
| SOL | paper_crowded_momentum_reversal_candidate | short_perp | 2500 | paper_execution_probe | 47.76 | 10.18 | 37.59 | 0.15 | 923704 | 0.0027 | public book does not obviously block a small paper probe |
| NEAR | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 46.99 | 11.51 | 35.48 | 1.47 | 59112 | 0.0042 | public book does not obviously block a small paper probe |
| NEAR | paper_crowded_momentum_reversal_candidate | short_perp | 1000 | paper_execution_probe | 46.99 | 11.64 | 35.35 | 1.47 | 59112 | 0.0169 | public book does not obviously block a small paper probe |
| NEAR | paper_crowded_momentum_reversal_candidate | short_perp | 2500 | paper_execution_probe | 46.99 | 11.89 | 35.10 | 1.47 | 59112 | 0.0423 | public book does not obviously block a small paper probe |
| ZRO | paper_extreme_funding_carry_candidate | short_perp | 250 | paper_execution_probe | 46.84 | 14.23 | 32.60 | 4.09 | 17827 | 0.0140 | public book does not obviously block a small paper probe |
| ZRO | paper_extreme_funding_carry_candidate | short_perp | 1000 | paper_execution_probe | 46.84 | 14.65 | 32.18 | 4.09 | 17827 | 0.0561 | public book does not obviously block a small paper probe |
| ZRO | paper_extreme_funding_carry_candidate | short_perp | 2500 | paper_execution_probe | 46.84 | 15.49 | 31.34 | 4.09 | 17827 | 0.1402 | public book does not obviously block a small paper probe |

## Interpretation

`paper_execution_probe` means the 15m label's gross edge still survives rough current taker fees, spread, and visible-depth impact. It still excludes queue position, partial fills, stop behavior, and repeated adverse selection.
