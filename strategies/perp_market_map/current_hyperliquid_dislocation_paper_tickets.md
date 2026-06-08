# Current Hyperliquid Dislocation Paper Tickets

This is not a trade instruction. It converts current 15m-supported dislocation probes into paper observation and falsification tickets.

- generated tickets: `5`

| asset | side | notional | horizon | gross15 | cost | net15 | spread | depth10 | usage |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| WLD | short_perp | 250 | 15m_observation_then_1h_confirmation | 38.87 | 11.06 | 27.81 | 1.05 | 146401 | 0.0017 |
| ZEC | short_perp | 250 | 15m_observation_then_1h_confirmation | 42.41 | 16.67 | 25.75 | 6.66 | 392114 | 0.0006 |
| PENGU | short_perp | 250 | 15m_observation_then_1h_confirmation | 31.76 | 15.95 | 15.81 | 5.79 | 15233 | 0.0164 |
| FET | short_perp | 250 | 15m_observation_then_1h_confirmation | 26.44 | 14.33 | 12.11 | 4.25 | 31614 | 0.0079 |
| SUI | short_perp | 250 | 15m_observation_then_1h_confirmation | 16.55 | 13.85 | 2.70 | 3.83 | 207781 | 0.0012 |

## Required Observations

- Mark/index move over the paper horizon.
- Funding drift and whether the edge survives fees.
- Spread, side depth, and visible-depth usage at observation time.
- Fresh snapshot persistence before treating the lane as repeatable.
- The next 1h label after the source snapshot matures.

## Reject If

- The 1h label fails.
- A fresh snapshot no longer supports the lane.
- Conservative net falls below zero.
- Spread or depth worsens materially.
- A stronger conflicting lane dominates the same asset.
