# Current Hyperliquid Dislocation Forward Labels

This labels Hyperliquid dislocation candidates after rough taker fees, impact spread, and funding carry. It is still paper labeling, not a live fill or deployable strategy.

- rows: `151`
- covered 15m: `0`
- covered 1h: `0`
- covered 4h: `0`

| asset | status | side | score | cost bps | net15 | out15 | net1h | out1h | net4h | out4h |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- | ---: | --- |
| JTO | paper_crowded_momentum_continuation_candidate | long_perp | 31.7857 | 29.79 |  | pending_15m |  | pending_1h |  | pending_4h |
| ZEC | paper_crowded_momentum_continuation_candidate | long_perp | 29.9571 | 16.75 |  | pending_15m |  | pending_1h |  | pending_4h |
| STABLE | paper_extreme_funding_carry_candidate | long_perp | 28.5189 | 46.46 |  | pending_15m |  | pending_1h |  | pending_4h |
| JTO | paper_crowded_momentum_reversal_candidate | short_perp | 27.0179 | 29.79 |  | pending_15m |  | pending_1h |  | pending_4h |
| ZEC | paper_crowded_momentum_reversal_candidate | short_perp | 25.4635 | 16.75 |  | pending_15m |  | pending_1h |  | pending_4h |
| WLD | paper_crowded_momentum_continuation_candidate | long_perp | 23.7677 | 22.34 |  | pending_15m |  | pending_1h |  | pending_4h |
| STBL | paper_crowded_momentum_continuation_candidate | long_perp | 20.9109 | 37.90 |  | pending_15m |  | pending_1h |  | pending_4h |
| EIGEN | paper_crowded_momentum_continuation_candidate | long_perp | 20.4156 | 26.69 |  | pending_15m |  | pending_1h |  | pending_4h |
| WLD | paper_crowded_momentum_reversal_candidate | short_perp | 20.2026 | 22.34 |  | pending_15m |  | pending_1h |  | pending_4h |
| DASH | paper_crowded_momentum_continuation_candidate | long_perp | 19.2233 | 26.16 |  | pending_15m |  | pending_1h |  | pending_4h |
| MANTA | paper_crowded_momentum_continuation_candidate | short_perp | 18.9496 | 34.95 |  | pending_15m |  | pending_1h |  | pending_4h |
| TAO | paper_crowded_momentum_continuation_candidate | long_perp | 18.9470 | 20.47 |  | pending_15m |  | pending_1h |  | pending_4h |
| LINK | paper_crowded_momentum_continuation_candidate | long_perp | 18.7842 | 15.33 |  | pending_15m |  | pending_1h |  | pending_4h |
| STBL | paper_crowded_momentum_reversal_candidate | short_perp | 17.7742 | 37.90 |  | pending_15m |  | pending_1h |  | pending_4h |
| AERO | paper_crowded_momentum_continuation_candidate | long_perp | 17.7245 | 30.60 |  | pending_15m |  | pending_1h |  | pending_4h |
| NEAR | paper_crowded_momentum_continuation_candidate | long_perp | 17.7121 | 18.81 |  | pending_15m |  | pending_1h |  | pending_4h |
| PENGU | paper_crowded_momentum_continuation_candidate | long_perp | 17.6555 | 15.73 |  | pending_15m |  | pending_1h |  | pending_4h |
| LDO | paper_crowded_momentum_continuation_candidate | long_perp | 17.6199 | 20.31 |  | pending_15m |  | pending_1h |  | pending_4h |
| DYDX | paper_crowded_momentum_continuation_candidate | long_perp | 17.4000 | 38.30 |  | pending_15m |  | pending_1h |  | pending_4h |
| EIGEN | paper_crowded_momentum_reversal_candidate | short_perp | 17.3533 | 26.69 |  | pending_15m |  | pending_1h |  | pending_4h |
| MEGA | paper_crowded_momentum_continuation_candidate | long_perp | 17.3438 | 27.37 |  | pending_15m |  | pending_1h |  | pending_4h |
| FARTCOIN | paper_crowded_momentum_continuation_candidate | long_perp | 16.9453 | 22.14 |  | pending_15m |  | pending_1h |  | pending_4h |
| PURR | paper_crowded_momentum_continuation_candidate | long_perp | 16.8726 | 148.26 |  | pending_15m |  | pending_1h |  | pending_4h |
| PUMP | paper_crowded_momentum_continuation_candidate | long_perp | 16.3460 | 23.11 |  | pending_15m |  | pending_1h |  | pending_4h |
| DASH | paper_crowded_momentum_reversal_candidate | short_perp | 16.3398 | 26.16 |  | pending_15m |  | pending_1h |  | pending_4h |
| MANTA | paper_crowded_momentum_reversal_candidate | long_perp | 16.1072 | 34.95 |  | pending_15m |  | pending_1h |  | pending_4h |
| TAO | paper_crowded_momentum_reversal_candidate | short_perp | 16.1050 | 20.47 |  | pending_15m |  | pending_1h |  | pending_4h |
| LINK | paper_crowded_momentum_reversal_candidate | short_perp | 15.9666 | 15.33 |  | pending_15m |  | pending_1h |  | pending_4h |
| BIO | paper_crowded_momentum_continuation_candidate | long_perp | 15.8075 | 25.26 |  | pending_15m |  | pending_1h |  | pending_4h |
| ATOM | paper_crowded_momentum_continuation_candidate | long_perp | 15.3958 | 29.84 |  | pending_15m |  | pending_1h |  | pending_4h |

## Interpretation

Positive net means the candidate side beat rough taker costs plus the funding carry estimate over that horizon. Pending rows simply have not had enough elapsed time since the candidate snapshot.
