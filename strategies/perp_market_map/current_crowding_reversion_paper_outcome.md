# Current Crowding Reversion Paper Outcome

This labels depth-gated Hyperliquid carry-reversion probes after the same rough cost proxy used by the execution check. It is still a paper observation, not a live fill.

- rows: `6`
- covered 15m: `6`
- covered 1h: `6`

| entry | asset | action | size | cost bps | net15 bps | out15 | net1h bps | out1h |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| 2026-06-07T23:28:18.721031+00:00 | ZRO | short_carry_reversion_watch | 250 | 10.95 | 38.34 | paper_15m_win | -23.75 | paper_1h_loss |
| 2026-06-07T23:28:18.721031+00:00 | APEX | short_carry_reversion_watch | 250 | 22.78 | 34.84 | paper_15m_win | -17.82 | paper_1h_loss |
| 2026-06-07T23:28:18.721031+00:00 | ETHFI | short_carry_reversion_watch | 250 | 18.67 | 33.13 | paper_15m_win | 17.52 | paper_1h_win |
| 2026-06-07T23:28:18.721031+00:00 | GRIFFAIN | short_carry_reversion_watch | 250 | 18.44 | 18.00 | paper_15m_win | -46.39 | paper_1h_loss |
| 2026-06-07T23:28:18.721031+00:00 | DYDX | short_carry_reversion_watch | 250 | 19.23 | -63.57 | paper_15m_loss | 16.10 | paper_1h_win |
| 2026-06-07T23:28:18.721031+00:00 | XMR | short_carry_reversion_watch | 250 | 14.41 | -67.59 | paper_15m_loss | -5.87 | paper_1h_loss |

## Interpretation

`paper_15m_win` or `paper_1h_win` only means the price moved in the candidate direction after the rough cost proxy. It still excludes actual order placement, queue position, partial fills, funding timing, mark/index basis, and stop behavior.
