# Current Hyperliquid Dislocation Forward Labels

This labels Hyperliquid dislocation candidates after rough taker fees, impact spread, and funding carry. It is still paper labeling, not a live fill or deployable strategy.

- rows: `175`
- covered 15m: `175`
- covered 1h: `0`
- covered 4h: `0`

| asset | status | side | score | cost bps | net15 | out15 | net1h | out1h | net4h | out4h |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- | ---: | --- |
| BRETT | paper_mark_oracle_reversion_candidate | long_perp | 5.1791 | 52.15 | 40.93 | paper_15m_win |  | pending_1h |  | pending_4h |
| EIGEN | paper_crowded_momentum_reversal_candidate | short_perp | 18.2823 | 32.19 | 39.87 | paper_15m_win |  | pending_1h |  | pending_4h |
| ZRO | paper_extreme_funding_carry_candidate | short_perp | 12.3732 | 11.21 | 38.27 | paper_15m_win |  | pending_1h |  | pending_4h |
| MEGA | paper_crowded_momentum_continuation_candidate | long_perp | 17.6576 | 23.30 | 29.61 | paper_15m_win |  | pending_1h |  | pending_4h |
| WLD | paper_crowded_momentum_reversal_candidate | short_perp | 17.3968 | 17.89 | 25.70 | paper_15m_win |  | pending_1h |  | pending_4h |
| MON | paper_crowded_momentum_continuation_candidate | long_perp | 16.0933 | 27.67 | 23.02 | paper_15m_win |  | pending_1h |  | pending_4h |
| ONDO | paper_crowded_momentum_continuation_candidate | long_perp | 12.8645 | 12.26 | 18.30 | paper_15m_win |  | pending_1h |  | pending_4h |
| ADA | paper_crowded_momentum_reversal_candidate | short_perp | 10.3637 | 14.29 | 17.69 | paper_15m_win |  | pending_1h |  | pending_4h |
| JTO | paper_crowded_momentum_reversal_candidate | short_perp | 27.4106 | 24.10 | 16.43 | paper_15m_win |  | pending_1h |  | pending_4h |
| AIXBT | paper_crowded_momentum_reversal_candidate | short_perp | 9.7806 | 25.84 | 14.86 | paper_15m_win |  | pending_1h |  | pending_4h |
| ZEN | paper_crowded_momentum_reversal_candidate | short_perp | 13.9987 | 24.49 | 14.24 | paper_15m_win |  | pending_1h |  | pending_4h |
| HYPE | paper_crowded_momentum_continuation_candidate | long_perp | 13.0101 | 10.89 | 14.23 | paper_15m_win |  | pending_1h |  | pending_4h |
| MANTA | paper_crowded_momentum_continuation_candidate | short_perp | 19.1774 | 25.70 | 13.57 | paper_15m_win |  | pending_1h |  | pending_4h |
| TAO | paper_crowded_momentum_continuation_candidate | long_perp | 19.2983 | 15.81 | 11.35 | paper_15m_win |  | pending_1h |  | pending_4h |
| kLUNC | paper_crowded_momentum_reversal_candidate | short_perp | 10.6781 | 21.98 | 10.87 | paper_15m_win |  | pending_1h |  | pending_4h |
| TIA | paper_crowded_momentum_reversal_candidate | short_perp | 10.1938 | 26.36 | 10.74 | paper_15m_win |  | pending_1h |  | pending_4h |
| DASH | paper_crowded_momentum_reversal_candidate | short_perp | 19.5613 | 28.39 | 9.27 | paper_15m_win |  | pending_1h |  | pending_4h |
| PNUT | paper_crowded_momentum_reversal_candidate | short_perp | 8.1595 | 26.64 | 9.03 | paper_15m_win |  | pending_1h |  | pending_4h |
| XRP | paper_crowded_momentum_reversal_candidate | short_perp | 11.0050 | 11.67 | 8.17 | paper_15m_win |  | pending_1h |  | pending_4h |
| CRV | paper_crowded_momentum_reversal_candidate | short_perp | 10.7382 | 18.13 | 7.29 | paper_15m_win |  | pending_1h |  | pending_4h |
| XPL | paper_crowded_momentum_continuation_candidate | long_perp | 12.2643 | 25.13 | 6.59 | paper_15m_win |  | pending_1h |  | pending_4h |
| SOL | paper_crowded_momentum_reversal_candidate | short_perp | 11.4146 | 10.21 | 6.19 | paper_15m_win |  | pending_1h |  | pending_4h |
| BNB | paper_crowded_momentum_reversal_candidate | short_perp | 11.6185 | 12.94 | 5.44 | paper_15m_win |  | pending_1h |  | pending_4h |
| NEAR | paper_crowded_momentum_continuation_candidate | long_perp | 17.1705 | 20.86 | 4.54 | paper_15m_win |  | pending_1h |  | pending_4h |
| LDO | paper_crowded_momentum_reversal_candidate | short_perp | 15.5869 | 21.77 | 4.32 | paper_15m_win |  | pending_1h |  | pending_4h |
| LINK | paper_crowded_momentum_reversal_candidate | short_perp | 16.0715 | 15.08 | 3.55 | paper_15m_win |  | pending_1h |  | pending_4h |
| kBONK | paper_crowded_momentum_reversal_candidate | short_perp | 10.2009 | 19.03 | 3.53 | paper_15m_win |  | pending_1h |  | pending_4h |
| FIL | paper_crowded_momentum_reversal_candidate | short_perp | 11.2852 | 23.06 | 1.69 | paper_15m_win |  | pending_1h |  | pending_4h |
| VIRTUAL | paper_crowded_momentum_reversal_candidate | short_perp | 12.2522 | 19.87 | 1.32 | paper_15m_win |  | pending_1h |  | pending_4h |
| DOGE | paper_crowded_momentum_reversal_candidate | short_perp | 10.4145 | 12.56 | 1.16 | paper_15m_win |  | pending_1h |  | pending_4h |

## Interpretation

Positive net means the candidate side beat rough taker costs plus the funding carry estimate over that horizon. Pending rows simply have not had enough elapsed time since the candidate snapshot.
