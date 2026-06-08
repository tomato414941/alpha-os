# Current Hyperliquid Dislocation Forward Labels

This labels Hyperliquid dislocation candidates after rough taker fees, impact spread, and funding carry. It is still paper labeling, not a live fill or deployable strategy.

- rows: `131`
- covered 15m: `131`
- covered 1h: `0`
- covered 4h: `0`

| asset | status | side | score | cost bps | net15 | out15 | net1h | out1h | net4h | out4h |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- | ---: | --- |
| BABY | paper_mark_oracle_reversion_candidate | long_perp | 5.0602 | 41.24 | 378.74 | paper_15m_win |  | pending_1h |  | pending_4h |
| BABY | paper_extreme_funding_carry_candidate | long_perp | 4.4180 | 41.24 | 378.74 | paper_15m_win |  | pending_1h |  | pending_4h |
| AERO | paper_crowded_momentum_continuation_candidate | long_perp | 17.6455 | 29.72 | 134.93 | paper_15m_win |  | pending_1h |  | pending_4h |
| AERO | paper_mark_oracle_reversion_candidate | long_perp | 13.0405 | 29.72 | 134.93 | paper_15m_win |  | pending_1h |  | pending_4h |
| AERO | paper_extreme_funding_carry_candidate | long_perp | 11.8885 | 29.72 | 134.93 | paper_15m_win |  | pending_1h |  | pending_4h |
| EIGEN | paper_crowded_momentum_continuation_candidate | long_perp | 19.5575 | 32.27 | 122.14 | paper_15m_win |  | pending_1h |  | pending_4h |
| EIGEN | paper_mark_oracle_reversion_candidate | long_perp | 7.9068 | 32.27 | 122.14 | paper_15m_win |  | pending_1h |  | pending_4h |
| HYPER | paper_mark_oracle_reversion_candidate | long_perp | 4.6025 | 40.29 | 103.13 | paper_15m_win |  | pending_1h |  | pending_4h |
| JTO | paper_crowded_momentum_continuation_candidate | long_perp | 32.1545 | 21.24 | 101.32 | paper_15m_win |  | pending_1h |  | pending_4h |
| JTO | paper_extreme_funding_carry_candidate | long_perp | 3.5951 | 21.24 | 101.32 | paper_15m_win |  | pending_1h |  | pending_4h |
| NEAR | paper_crowded_momentum_continuation_candidate | long_perp | 17.0863 | 19.17 | 87.57 | paper_15m_win |  | pending_1h |  | pending_4h |
| ONDO | paper_crowded_momentum_continuation_candidate | long_perp | 11.9795 | 19.63 | 85.65 | paper_15m_win |  | pending_1h |  | pending_4h |
| VVV | paper_crowded_momentum_continuation_candidate | long_perp | 12.4392 | 20.36 | 85.03 | paper_15m_win |  | pending_1h |  | pending_4h |
| PENGU | paper_crowded_momentum_continuation_candidate | long_perp | 15.6806 | 15.81 | 72.36 | paper_15m_win |  | pending_1h |  | pending_4h |
| INIT | paper_mark_oracle_reversion_candidate | long_perp | 5.5218 | 37.98 | 69.34 | paper_15m_win |  | pending_1h |  | pending_4h |
| VIRTUAL | paper_crowded_momentum_continuation_candidate | long_perp | 13.3440 | 16.57 | 68.41 | paper_15m_win |  | pending_1h |  | pending_4h |
| PUMP | paper_crowded_momentum_continuation_candidate | long_perp | 16.1234 | 16.56 | 68.28 | paper_15m_win |  | pending_1h |  | pending_4h |
| TRUMP | paper_mark_oracle_reversion_candidate | long_perp | 9.0503 | 18.89 | 65.76 | paper_15m_win |  | pending_1h |  | pending_4h |
| TRUMP | paper_extreme_funding_carry_candidate | long_perp | 7.7528 | 18.89 | 65.76 | paper_15m_win |  | pending_1h |  | pending_4h |
| PYTH | paper_mark_oracle_reversion_candidate | long_perp | 5.0218 | 31.70 | 60.95 | paper_15m_win |  | pending_1h |  | pending_4h |
| IMX | paper_crowded_momentum_continuation_candidate | long_perp | 13.7733 | 43.07 | 58.69 | paper_15m_win |  | pending_1h |  | pending_4h |
| LDO | paper_crowded_momentum_continuation_candidate | long_perp | 17.5058 | 19.94 | 57.05 | paper_15m_win |  | pending_1h |  | pending_4h |
| ZORA | paper_mark_oracle_reversion_candidate | long_perp | 8.7764 | 39.44 | 55.80 | paper_15m_win |  | pending_1h |  | pending_4h |
| ZORA | paper_extreme_funding_carry_candidate | long_perp | 6.1731 | 39.44 | 55.80 | paper_15m_win |  | pending_1h |  | pending_4h |
| FIL | paper_crowded_momentum_continuation_candidate | long_perp | 11.7939 | 24.95 | 54.64 | paper_15m_win |  | pending_1h |  | pending_4h |
| POPCAT | paper_mark_oracle_reversion_candidate | long_perp | 7.1246 | 21.63 | 53.41 | paper_15m_win |  | pending_1h |  | pending_4h |
| TAO | paper_crowded_momentum_continuation_candidate | long_perp | 18.4357 | 18.92 | 52.82 | paper_15m_win |  | pending_1h |  | pending_4h |
| SUSHI | paper_mark_oracle_reversion_candidate | long_perp | 5.5481 | 30.36 | 52.59 | paper_15m_win |  | pending_1h |  | pending_4h |
| WLD | paper_crowded_momentum_continuation_candidate | long_perp | 17.5478 | 16.98 | 48.67 | paper_15m_win |  | pending_1h |  | pending_4h |
| SPX | paper_crowded_momentum_continuation_candidate | long_perp | 11.4227 | 30.31 | 46.96 | paper_15m_win |  | pending_1h |  | pending_4h |

## Interpretation

Positive net means the candidate side beat rough taker costs plus the funding carry estimate over that horizon. Pending rows simply have not had enough elapsed time since the candidate snapshot.
