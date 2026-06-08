# Current Hyperliquid Dislocation Forward Labels

This labels Hyperliquid dislocation candidates after rough taker fees, impact spread, and funding carry. It is still paper labeling, not a live fill or deployable strategy.

- rows: `197`
- covered 15m: `0`
- covered 1h: `0`
- covered 4h: `0`

| asset | status | side | score | cost bps | net15 | out15 | net1h | out1h | net4h | out4h |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- | ---: | --- |
| LAYER | paper_extreme_funding_carry_candidate | long_perp | 112.8941 | 55.98 |  | pending_15m |  | pending_1h |  | pending_4h |
| LAYER | paper_crowded_momentum_continuation_candidate | long_perp | 53.0087 | 55.98 |  | pending_15m |  | pending_1h |  | pending_4h |
| LAYER | paper_crowded_momentum_reversal_candidate | short_perp | 45.0574 | 55.98 |  | pending_15m |  | pending_1h |  | pending_4h |
| BIO | paper_crowded_momentum_continuation_candidate | long_perp | 31.3989 | 26.00 |  | pending_15m |  | pending_1h |  | pending_4h |
| MOVE | paper_crowded_momentum_continuation_candidate | long_perp | 30.0944 | 27.27 |  | pending_15m |  | pending_1h |  | pending_4h |
| WLD | paper_crowded_momentum_continuation_candidate | long_perp | 27.8915 | 17.27 |  | pending_15m |  | pending_1h |  | pending_4h |
| BIO | paper_crowded_momentum_reversal_candidate | short_perp | 26.6891 | 26.00 |  | pending_15m |  | pending_1h |  | pending_4h |
| MOVE | paper_crowded_momentum_reversal_candidate | short_perp | 25.5802 | 27.27 |  | pending_15m |  | pending_1h |  | pending_4h |
| WLD | paper_crowded_momentum_reversal_candidate | short_perp | 23.7078 | 17.27 |  | pending_15m |  | pending_1h |  | pending_4h |
| CHIP | paper_crowded_momentum_continuation_candidate | long_perp | 22.2200 | 21.31 |  | pending_15m |  | pending_1h |  | pending_4h |
| PUMP | paper_crowded_momentum_continuation_candidate | long_perp | 22.0425 | 22.14 |  | pending_15m |  | pending_1h |  | pending_4h |
| BERA | paper_crowded_momentum_continuation_candidate | long_perp | 21.7687 | 25.65 |  | pending_15m |  | pending_1h |  | pending_4h |
| ONDO | paper_crowded_momentum_continuation_candidate | long_perp | 21.3741 | 20.52 |  | pending_15m |  | pending_1h |  | pending_4h |
| kLUNC | paper_crowded_momentum_continuation_candidate | long_perp | 21.1299 | 23.16 |  | pending_15m |  | pending_1h |  | pending_4h |
| ATOM | paper_crowded_momentum_continuation_candidate | long_perp | 21.0029 | 29.52 |  | pending_15m |  | pending_1h |  | pending_4h |
| PURR | paper_crowded_momentum_continuation_candidate | long_perp | 20.8615 | 139.50 |  | pending_15m |  | pending_1h |  | pending_4h |
| HYPE | paper_crowded_momentum_continuation_candidate | long_perp | 19.4740 | 10.16 |  | pending_15m |  | pending_1h |  | pending_4h |
| NXPC | paper_crowded_momentum_continuation_candidate | long_perp | 19.3889 | 44.11 |  | pending_15m |  | pending_1h |  | pending_4h |
| LIT | paper_crowded_momentum_continuation_candidate | long_perp | 19.2707 | 23.34 |  | pending_15m |  | pending_1h |  | pending_4h |
| ZEC | paper_crowded_momentum_continuation_candidate | long_perp | 19.0516 | 14.22 |  | pending_15m |  | pending_1h |  | pending_4h |
| CHIP | paper_crowded_momentum_reversal_candidate | short_perp | 18.8870 | 21.31 |  | pending_15m |  | pending_1h |  | pending_4h |
| VVV | paper_crowded_momentum_continuation_candidate | long_perp | 18.8664 | 23.99 |  | pending_15m |  | pending_1h |  | pending_4h |
| PUMP | paper_crowded_momentum_reversal_candidate | short_perp | 18.7362 | 22.14 |  | pending_15m |  | pending_1h |  | pending_4h |
| NEAR | paper_crowded_momentum_continuation_candidate | long_perp | 18.6490 | 17.29 |  | pending_15m |  | pending_1h |  | pending_4h |
| BERA | paper_crowded_momentum_reversal_candidate | short_perp | 18.5034 | 25.65 |  | pending_15m |  | pending_1h |  | pending_4h |
| MELANIA | paper_crowded_momentum_continuation_candidate | long_perp | 18.2906 | 25.09 |  | pending_15m |  | pending_1h |  | pending_4h |
| ONDO | paper_crowded_momentum_reversal_candidate | short_perp | 18.1680 | 20.52 |  | pending_15m |  | pending_1h |  | pending_4h |
| MEGA | paper_crowded_momentum_continuation_candidate | long_perp | 17.9622 | 22.82 |  | pending_15m |  | pending_1h |  | pending_4h |
| kLUNC | paper_crowded_momentum_reversal_candidate | short_perp | 17.9604 | 23.16 |  | pending_15m |  | pending_1h |  | pending_4h |
| ATOM | paper_crowded_momentum_reversal_candidate | short_perp | 17.8525 | 29.52 |  | pending_15m |  | pending_1h |  | pending_4h |

## Interpretation

Positive net means the candidate side beat rough taker costs plus the funding carry estimate over that horizon. Pending rows simply have not had enough elapsed time since the candidate snapshot.
