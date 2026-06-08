# Current OKX Liquidation Paper Outcome

This joins paper-gate rows to monitor forward labels. It measures the paper result after the same conservative cost proxy used by the gate. It is still a retrospective observation, not a live fill.

| event | asset | action | dir | size USD | cost bps | net15 bps | out15 | net1h bps | out1h |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- |

## Interpretation

`paper_15m_win` means the event label stayed positive after the rough cost proxy. A deployable strategy still needs fresh-event repetition, live fills, risk limits, and a rule for skipping crowded or stale events.
