# Current Hyperliquid Dislocation Execution Check

This applies a current public-book gate to 15m-supported Hyperliquid dislocation labels. It is still not a fill model.

- rows: `15`
- paper execution probes: `15`

| asset | status | side | size | gate | gross15 | cost | conservative15 | spread | depth10 | usage | reason |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ZEC | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 42.41 | 11.18 | 31.24 | 1.15 | 89264 | 0.0028 | public book does not obviously block a small paper probe |
| ZEC | paper_crowded_momentum_reversal_candidate | short_perp | 1000 | paper_execution_probe | 42.41 | 11.26 | 31.15 | 1.15 | 89264 | 0.0112 | public book does not obviously block a small paper probe |
| ZEC | paper_crowded_momentum_reversal_candidate | short_perp | 2500 | paper_execution_probe | 42.41 | 11.43 | 30.98 | 1.15 | 89264 | 0.0280 | public book does not obviously block a small paper probe |
| WLD | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 38.87 | 11.62 | 27.25 | 1.49 | 18351 | 0.0136 | public book does not obviously block a small paper probe |
| WLD | paper_crowded_momentum_reversal_candidate | short_perp | 1000 | paper_execution_probe | 38.87 | 12.03 | 26.84 | 1.49 | 18351 | 0.0545 | public book does not obviously block a small paper probe |
| WLD | paper_crowded_momentum_reversal_candidate | short_perp | 2500 | paper_execution_probe | 38.87 | 12.85 | 26.02 | 1.49 | 18351 | 0.1362 | public book does not obviously block a small paper probe |
| PENGU | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 31.76 | 18.85 | 12.91 | 8.72 | 18772 | 0.0133 | public book does not obviously block a small paper probe |
| PENGU | paper_crowded_momentum_reversal_candidate | short_perp | 1000 | paper_execution_probe | 31.76 | 19.25 | 12.51 | 8.72 | 18772 | 0.0533 | public book does not obviously block a small paper probe |
| PENGU | paper_crowded_momentum_reversal_candidate | short_perp | 2500 | paper_execution_probe | 31.76 | 20.05 | 11.71 | 8.72 | 18772 | 0.1332 | public book does not obviously block a small paper probe |
| FET | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 26.44 | 15.81 | 10.63 | 5.67 | 17357 | 0.0144 | public book does not obviously block a small paper probe |
| FET | paper_crowded_momentum_reversal_candidate | short_perp | 1000 | paper_execution_probe | 26.44 | 16.24 | 10.20 | 5.67 | 17357 | 0.0576 | public book does not obviously block a small paper probe |
| FET | paper_crowded_momentum_reversal_candidate | short_perp | 2500 | paper_execution_probe | 26.44 | 17.11 | 9.34 | 5.67 | 17357 | 0.1440 | public book does not obviously block a small paper probe |
| SUI | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 16.55 | 11.36 | 5.19 | 1.32 | 71564 | 0.0035 | public book does not obviously block a small paper probe |
| SUI | paper_crowded_momentum_reversal_candidate | short_perp | 1000 | paper_execution_probe | 16.55 | 11.46 | 5.09 | 1.32 | 71564 | 0.0140 | public book does not obviously block a small paper probe |
| SUI | paper_crowded_momentum_reversal_candidate | short_perp | 2500 | paper_execution_probe | 16.55 | 11.67 | 4.88 | 1.32 | 71564 | 0.0349 | public book does not obviously block a small paper probe |

## Interpretation

`paper_execution_probe` means the 15m label's gross edge still survives rough current taker fees, spread, and visible-depth impact. It still excludes queue position, partial fills, stop behavior, and repeated adverse selection.
