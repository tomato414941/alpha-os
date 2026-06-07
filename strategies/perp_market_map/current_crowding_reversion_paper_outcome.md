# Current Crowding Reversion Paper Outcome

This labels depth-gated Hyperliquid carry-reversion probes after the same rough cost proxy used by the execution check. It is still a paper observation, not a live fill.

- rows: `6`
- covered 15m: `0`
- covered 1h: `0`

| entry | asset | action | size | cost bps | net15 bps | out15 | net1h bps | out1h |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| 2026-06-07T23:28:18.721031+00:00 | DYDX | short_carry_reversion_watch | 250 | 19.23 |  | pending_15m |  | pending_1h |
| 2026-06-07T23:28:18.721031+00:00 | ZRO | short_carry_reversion_watch | 250 | 10.95 |  | pending_15m |  | pending_1h |
| 2026-06-07T23:28:18.721031+00:00 | XMR | short_carry_reversion_watch | 250 | 14.41 |  | pending_15m |  | pending_1h |
| 2026-06-07T23:28:18.721031+00:00 | ETHFI | short_carry_reversion_watch | 250 | 18.67 |  | pending_15m |  | pending_1h |
| 2026-06-07T23:28:18.721031+00:00 | GRIFFAIN | short_carry_reversion_watch | 250 | 18.44 |  | pending_15m |  | pending_1h |
| 2026-06-07T23:28:18.721031+00:00 | APEX | short_carry_reversion_watch | 250 | 22.78 |  | pending_15m |  | pending_1h |

## Interpretation

`paper_15m_win` or `paper_1h_win` only means the price moved in the candidate direction after the rough cost proxy. It still excludes actual order placement, queue position, partial fills, funding timing, mark/index basis, and stop behavior.
