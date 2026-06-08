# Current Hyperliquid Dislocation Forward Labels

This labels Hyperliquid dislocation candidates after rough taker fees, impact spread, and funding carry. It is still paper labeling, not a live fill or deployable strategy.

- rows: `124`
- covered 15m: `0`
- covered 1h: `0`
- covered 4h: `0`

| asset | status | side | score | cost bps | net15 | out15 | net1h | out1h | net4h | out4h |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- | ---: | --- |
| STABLE | paper_extreme_funding_carry_candidate | long_perp | 36.5990 | 34.15 |  | pending_15m |  | pending_1h |  | pending_4h |
| LAYER | paper_extreme_funding_carry_candidate | long_perp | 30.0702 | 49.07 |  | pending_15m |  | pending_1h |  | pending_4h |
| MOVE | paper_crowded_momentum_continuation_candidate | long_perp | 27.6144 | 42.06 |  | pending_15m |  | pending_1h |  | pending_4h |
| MOVE | paper_crowded_momentum_reversal_candidate | short_perp | 23.4722 | 42.06 |  | pending_15m |  | pending_1h |  | pending_4h |
| BIO | paper_crowded_momentum_continuation_candidate | long_perp | 22.5058 | 27.16 |  | pending_15m |  | pending_1h |  | pending_4h |
| PURR | paper_crowded_momentum_continuation_candidate | long_perp | 22.4583 | 191.11 |  | pending_15m |  | pending_1h |  | pending_4h |
| CC | paper_crowded_momentum_continuation_candidate | short_perp | 19.4722 | 45.88 |  | pending_15m |  | pending_1h |  | pending_4h |
| LAYER | paper_crowded_momentum_continuation_candidate | long_perp | 19.1848 | 49.07 |  | pending_15m |  | pending_1h |  | pending_4h |
| BIO | paper_crowded_momentum_reversal_candidate | short_perp | 19.1299 | 27.16 |  | pending_15m |  | pending_1h |  | pending_4h |
| PURR | paper_crowded_momentum_reversal_candidate | short_perp | 19.0896 | 191.11 |  | pending_15m |  | pending_1h |  | pending_4h |
| MORPHO | paper_crowded_momentum_continuation_candidate | long_perp | 18.6888 | 30.24 |  | pending_15m |  | pending_1h |  | pending_4h |
| HYPE | paper_crowded_momentum_continuation_candidate | long_perp | 18.4980 | 11.14 |  | pending_15m |  | pending_1h |  | pending_4h |
| NXPC | paper_crowded_momentum_continuation_candidate | long_perp | 18.1120 | 26.99 |  | pending_15m |  | pending_1h |  | pending_4h |
| LIT | paper_crowded_momentum_continuation_candidate | long_perp | 17.4707 | 25.69 |  | pending_15m |  | pending_1h |  | pending_4h |
| MEGA | paper_crowded_momentum_continuation_candidate | long_perp | 17.4415 | 35.85 |  | pending_15m |  | pending_1h |  | pending_4h |
| CHIP | paper_crowded_momentum_continuation_candidate | long_perp | 16.9456 | 24.72 |  | pending_15m |  | pending_1h |  | pending_4h |
| CC | paper_crowded_momentum_reversal_candidate | long_perp | 16.5514 | 45.88 |  | pending_15m |  | pending_1h |  | pending_4h |
| ATOM | paper_crowded_momentum_continuation_candidate | long_perp | 16.4499 | 30.34 |  | pending_15m |  | pending_1h |  | pending_4h |
| IMX | paper_crowded_momentum_continuation_candidate | long_perp | 16.3110 | 53.20 |  | pending_15m |  | pending_1h |  | pending_4h |
| LAYER | paper_crowded_momentum_reversal_candidate | short_perp | 16.3071 | 49.07 |  | pending_15m |  | pending_1h |  | pending_4h |
| BERA | paper_crowded_momentum_continuation_candidate | long_perp | 16.2220 | 31.03 |  | pending_15m |  | pending_1h |  | pending_4h |
| MORPHO | paper_crowded_momentum_reversal_candidate | short_perp | 15.8855 | 30.24 |  | pending_15m |  | pending_1h |  | pending_4h |
| HYPE | paper_crowded_momentum_reversal_candidate | short_perp | 15.7233 | 11.14 |  | pending_15m |  | pending_1h |  | pending_4h |
| NEAR | paper_crowded_momentum_continuation_candidate | long_perp | 15.5409 | 16.59 |  | pending_15m |  | pending_1h |  | pending_4h |
| NXPC | paper_crowded_momentum_reversal_candidate | short_perp | 15.3952 | 26.99 |  | pending_15m |  | pending_1h |  | pending_4h |
| PUMP | paper_crowded_momentum_continuation_candidate | long_perp | 15.2792 | 22.34 |  | pending_15m |  | pending_1h |  | pending_4h |
| LIT | paper_crowded_momentum_reversal_candidate | short_perp | 14.8501 | 25.69 |  | pending_15m |  | pending_1h |  | pending_4h |
| MEGA | paper_crowded_momentum_reversal_candidate | short_perp | 14.8252 | 35.85 |  | pending_15m |  | pending_1h |  | pending_4h |
| CHIP | paper_crowded_momentum_reversal_candidate | short_perp | 14.4037 | 24.72 |  | pending_15m |  | pending_1h |  | pending_4h |
| KAS | paper_crowded_momentum_continuation_candidate | long_perp | 14.0191 | 45.04 |  | pending_15m |  | pending_1h |  | pending_4h |

## Interpretation

Positive net means the candidate side beat rough taker costs plus the funding carry estimate over that horizon. Pending rows simply have not had enough elapsed time since the candidate snapshot.
