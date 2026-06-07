# Current Crowding Reversion Paper Outcome

This labels depth-gated Hyperliquid carry-reversion probes after the same rough cost proxy used by the execution check. It is still a paper observation, not a live fill.

- rows: `6`
- covered 15m: `0`
- covered 1h: `0`

| entry | asset | action | size | cost bps | net15 bps | out15 | net1h bps | out1h |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| 2026-06-07T23:16:59.958447+00:00 | DYDX | short_carry_reversion_watch | 250 | 17.81 |  | pending_15m |  | pending_1h |
| 2026-06-07T23:16:59.958447+00:00 | ZRO | short_carry_reversion_watch | 250 | 18.30 |  | pending_15m |  | pending_1h |
| 2026-06-07T23:16:59.958447+00:00 | ETHFI | short_carry_reversion_watch | 250 | 11.93 |  | pending_15m |  | pending_1h |
| 2026-06-07T23:16:59.958447+00:00 | XMR | short_carry_reversion_watch | 250 | 12.17 |  | pending_15m |  | pending_1h |
| 2026-06-07T23:16:59.958447+00:00 | CFX | short_carry_reversion_watch | 250 | 14.53 |  | pending_15m |  | pending_1h |
| 2026-06-07T23:16:59.958447+00:00 | HEMI | short_carry_reversion_watch | 250 | 18.38 |  | pending_15m |  | pending_1h |

## Interpretation

`paper_15m_win` or `paper_1h_win` only means the price moved in the candidate direction after the rough cost proxy. It still excludes actual order placement, queue position, partial fills, funding timing, mark/index basis, and stop behavior.
