# Current Hyperliquid Dislocation Execution Check

This applies a current public-book gate to 15m-supported Hyperliquid dislocation labels. It is still not a fill model.

- rows: `15`
- paper execution probes: `15`

| asset | status | side | size | gate | gross15 | cost | conservative15 | spread | depth10 | usage | reason |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| WLD | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 38.87 | 11.06 | 27.81 | 1.05 | 146401 | 0.0017 | public book does not obviously block a small paper probe |
| WLD | paper_crowded_momentum_reversal_candidate | short_perp | 1000 | paper_execution_probe | 38.87 | 11.12 | 27.75 | 1.05 | 146401 | 0.0068 | public book does not obviously block a small paper probe |
| WLD | paper_crowded_momentum_reversal_candidate | short_perp | 2500 | paper_execution_probe | 38.87 | 11.22 | 27.65 | 1.05 | 146401 | 0.0171 | public book does not obviously block a small paper probe |
| ZEC | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 42.41 | 16.67 | 25.75 | 6.66 | 392114 | 0.0006 | public book does not obviously block a small paper probe |
| ZEC | paper_crowded_momentum_reversal_candidate | short_perp | 1000 | paper_execution_probe | 42.41 | 16.68 | 25.73 | 6.66 | 392114 | 0.0026 | public book does not obviously block a small paper probe |
| ZEC | paper_crowded_momentum_reversal_candidate | short_perp | 2500 | paper_execution_probe | 42.41 | 16.72 | 25.69 | 6.66 | 392114 | 0.0064 | public book does not obviously block a small paper probe |
| PENGU | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 31.76 | 15.95 | 15.81 | 5.79 | 15233 | 0.0164 | public book does not obviously block a small paper probe |
| PENGU | paper_crowded_momentum_reversal_candidate | short_perp | 1000 | paper_execution_probe | 31.76 | 16.44 | 15.32 | 5.79 | 15233 | 0.0656 | public book does not obviously block a small paper probe |
| PENGU | paper_crowded_momentum_reversal_candidate | short_perp | 2500 | paper_execution_probe | 31.76 | 17.43 | 14.34 | 5.79 | 15233 | 0.1641 | public book does not obviously block a small paper probe |
| FET | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 26.44 | 14.33 | 12.11 | 4.25 | 31614 | 0.0079 | public book does not obviously block a small paper probe |
| FET | paper_crowded_momentum_reversal_candidate | short_perp | 1000 | paper_execution_probe | 26.44 | 14.57 | 11.88 | 4.25 | 31614 | 0.0316 | public book does not obviously block a small paper probe |
| FET | paper_crowded_momentum_reversal_candidate | short_perp | 2500 | paper_execution_probe | 26.44 | 15.04 | 11.40 | 4.25 | 31614 | 0.0791 | public book does not obviously block a small paper probe |
| SUI | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 16.55 | 13.85 | 2.70 | 3.83 | 207781 | 0.0012 | public book does not obviously block a small paper probe |
| SUI | paper_crowded_momentum_reversal_candidate | short_perp | 1000 | paper_execution_probe | 16.55 | 13.88 | 2.67 | 3.83 | 207781 | 0.0048 | public book does not obviously block a small paper probe |
| SUI | paper_crowded_momentum_reversal_candidate | short_perp | 2500 | paper_execution_probe | 16.55 | 13.96 | 2.60 | 3.83 | 207781 | 0.0120 | public book does not obviously block a small paper probe |

## Interpretation

`paper_execution_probe` means the 15m label's gross edge still survives rough current taker fees, spread, and visible-depth impact. It still excludes queue position, partial fills, stop behavior, and repeated adverse selection.
