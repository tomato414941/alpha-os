# Current Hyperliquid Dislocation Execution Check

This applies a current public-book gate to 15m-supported Hyperliquid dislocation labels. It is still not a fill model.

- rows: `60`
- paper execution probes: `4`

| asset | status | side | size | gate | gross15 | net15 | out1h | net1h | spread | depth10 | usage | reason |
| --- | --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| ZEC | paper_crowded_momentum_reversal_candidate | short_perp | 250 | paper_execution_probe | 40.01 | 27.41 | paper_1h_win | 229.69 | 2.56 | 67915 | 0.0037 | public book does not obviously block a small paper probe |
| ZEC | paper_crowded_momentum_reversal_candidate | short_perp | 1000 | paper_execution_probe | 40.01 | 27.30 | paper_1h_win | 229.58 | 2.56 | 67915 | 0.0147 | public book does not obviously block a small paper probe |
| ZEC | paper_crowded_momentum_reversal_candidate | short_perp | 2500 | paper_execution_probe | 40.01 | 27.08 | paper_1h_win | 229.36 | 2.56 | 67915 | 0.0368 | public book does not obviously block a small paper probe |
| MANTA | paper_crowded_momentum_continuation_candidate | short_perp | 250 | paper_execution_probe | 33.78 | 17.77 | paper_1h_win | 129.51 | 5.27 | 3332 | 0.0750 | public book does not obviously block a small paper probe |
| BABY | paper_mark_oracle_reversion_candidate | long_perp | 250 | too_large_for_visible_depth | 419.99 | 404.82 | paper_1h_win | 585.97 | 1.72 | 726 | 0.3443 | candidate size uses too much visible near-touch depth |
| BABY | paper_extreme_funding_carry_candidate | long_perp | 250 | too_large_for_visible_depth | 419.99 | 404.82 | paper_1h_win | 585.97 | 1.72 | 726 | 0.3443 | candidate size uses too much visible near-touch depth |
| BABY | paper_mark_oracle_reversion_candidate | long_perp | 1000 | too_large_for_visible_depth | 419.99 | 398.27 | paper_1h_win | 579.41 | 1.72 | 726 | 1.3774 | candidate size uses too much visible near-touch depth |
| BABY | paper_extreme_funding_carry_candidate | long_perp | 1000 | too_large_for_visible_depth | 419.99 | 398.27 | paper_1h_win | 579.41 | 1.72 | 726 | 1.3774 | candidate size uses too much visible near-touch depth |
| BABY | paper_mark_oracle_reversion_candidate | long_perp | 2500 | too_large_for_visible_depth | 419.99 | 398.27 | paper_1h_win | 579.41 | 1.72 | 726 | 3.4435 | candidate size uses too much visible near-touch depth |
| BABY | paper_extreme_funding_carry_candidate | long_perp | 2500 | too_large_for_visible_depth | 419.99 | 398.27 | paper_1h_win | 579.41 | 1.72 | 726 | 3.4435 | candidate size uses too much visible near-touch depth |
| MANTA | paper_crowded_momentum_continuation_candidate | short_perp | 1000 | too_large_for_visible_depth | 33.78 | 15.52 | paper_1h_win | 127.26 | 5.27 | 3332 | 0.3001 | candidate size uses too much visible near-touch depth |
| MANTA | paper_crowded_momentum_continuation_candidate | short_perp | 2500 | too_large_for_visible_depth | 33.78 | 11.02 | paper_1h_win | 122.76 | 5.27 | 3332 | 0.7502 | candidate size uses too much visible near-touch depth |
| EIGEN | paper_crowded_momentum_continuation_candidate | long_perp | 250 | failed_1h_confirmation | 154.41 | 137.97 | paper_1h_loss | -60.69 | 5.53 | 2751 | 0.0909 | 15m edge reversed by the matured 1h label |
| EIGEN | paper_mark_oracle_reversion_candidate | long_perp | 250 | failed_1h_confirmation | 154.41 | 137.97 | paper_1h_loss | -60.69 | 5.53 | 2751 | 0.0909 | 15m edge reversed by the matured 1h label |
| EIGEN | paper_crowded_momentum_continuation_candidate | long_perp | 1000 | failed_1h_confirmation | 154.41 | 135.25 | paper_1h_loss | -63.41 | 5.53 | 2751 | 0.3634 | 15m edge reversed by the matured 1h label |
| EIGEN | paper_mark_oracle_reversion_candidate | long_perp | 1000 | failed_1h_confirmation | 154.41 | 135.25 | paper_1h_loss | -63.41 | 5.53 | 2751 | 0.3634 | 15m edge reversed by the matured 1h label |
| EIGEN | paper_crowded_momentum_continuation_candidate | long_perp | 2500 | failed_1h_confirmation | 154.41 | 129.80 | paper_1h_loss | -68.86 | 5.53 | 2751 | 0.9086 | 15m edge reversed by the matured 1h label |
| EIGEN | paper_mark_oracle_reversion_candidate | long_perp | 2500 | failed_1h_confirmation | 154.41 | 129.80 | paper_1h_loss | -68.86 | 5.53 | 2751 | 0.9086 | 15m edge reversed by the matured 1h label |
| NEAR | paper_crowded_momentum_continuation_candidate | long_perp | 250 | failed_1h_confirmation | 106.74 | 96.22 | paper_1h_loss | -78.58 | 0.49 | 100568 | 0.0025 | 15m edge reversed by the matured 1h label |
| NEAR | paper_crowded_momentum_continuation_candidate | long_perp | 1000 | failed_1h_confirmation | 106.74 | 96.15 | paper_1h_loss | -78.66 | 0.49 | 100568 | 0.0099 | 15m edge reversed by the matured 1h label |

## Interpretation

`paper_execution_probe` means the 15m label's gross edge still survives rough current taker fees, spread, and visible-depth impact. It still excludes queue position, partial fills, stop behavior, and repeated adverse selection.
