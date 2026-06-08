# Current Hyperliquid Dislocation Forward Labels

This labels Hyperliquid dislocation candidates after rough taker fees, impact spread, and funding carry. It is still paper labeling, not a live fill or deployable strategy.

- rows: `10`
- covered 15m: `10`
- covered 1h: `0`
- covered 4h: `0`

| asset | status | side | score | cost bps | net15 | out15 | net1h | out1h | net4h | out4h |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- | ---: | --- |
| LDO | paper_crowded_momentum_continuation_candidate | long_perp | 17.5058 | 19.94 | 52.65 | paper_15m_win |  | pending_1h |  | pending_4h |
| MON | paper_crowded_momentum_continuation_candidate | long_perp | 16.0807 | 23.03 | 46.54 | paper_15m_win |  | pending_1h |  | pending_4h |
| kLUNC | paper_crowded_momentum_continuation_candidate | long_perp | 13.0471 | 24.72 | 5.28 | paper_15m_win |  | pending_1h |  | pending_4h |
| DASH | paper_crowded_momentum_continuation_candidate | long_perp | 18.8049 | 22.74 | 2.62 | paper_15m_win |  | pending_1h |  | pending_4h |
| DASH | paper_crowded_momentum_reversal_candidate | short_perp | 15.9841 | 22.74 | -48.10 | paper_15m_loss |  | pending_1h |  | pending_4h |
| LDO | paper_crowded_momentum_reversal_candidate | short_perp | 14.8800 | 19.94 | -92.54 | paper_15m_loss |  | pending_1h |  | pending_4h |
| MON | paper_crowded_momentum_reversal_candidate | short_perp | 13.6686 | 23.03 | -92.60 | paper_15m_loss |  | pending_1h |  | pending_4h |
| PURR | paper_crowded_momentum_continuation_candidate | long_perp | 20.6119 | 135.15 | -94.52 | paper_15m_loss |  | pending_1h |  | pending_4h |
| PURR | paper_crowded_momentum_reversal_candidate | short_perp | 17.5201 | 135.15 | -175.77 | paper_15m_loss |  | pending_1h |  | pending_4h |
| PURR | paper_mark_oracle_reversion_candidate | short_perp | 14.5394 | 135.15 | -175.77 | paper_15m_loss |  | pending_1h |  | pending_4h |

## Interpretation

Positive net means the candidate side beat rough taker costs plus the funding carry estimate over that horizon. Pending rows simply have not had enough elapsed time since the candidate snapshot.
