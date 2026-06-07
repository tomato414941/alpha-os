# Current OKX Liquidation Paper Outcome

This joins paper-gate rows to monitor forward labels. It measures the paper result after the same conservative cost proxy used by the gate. It is still a retrospective observation, not a live fill.

| event | asset | action | dir | size USD | cost bps | net15 bps | out15 | net1h bps | out1h |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| 2026-06-07T15:26:20.101000+00:00 | ONDO | short_liquidation_squeeze_watch | long | 100 | 12.95 | 61.57 | paper_15m_win | 7.11 | paper_1h_win |

## Interpretation

`paper_15m_win` means the event label stayed positive after the rough cost proxy. A deployable strategy still needs fresh-event repetition, live fills, risk limits, and a rule for skipping crowded or stale events.
