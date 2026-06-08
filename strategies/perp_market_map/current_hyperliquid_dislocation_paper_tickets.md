# Current Hyperliquid Dislocation Paper Tickets

This is not a trade instruction. It converts current 15m-supported dislocation probes into paper observation and falsification tickets.

- generated tickets: `5`

| asset | side | notional | horizon | gross15 | cost | net15 | spread | depth10 | usage |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ZEC | short_perp | 250 | 15m_observation_then_1h_confirmation | 42.41 | 11.18 | 31.24 | 1.15 | 89264 | 0.0028 |
| WLD | short_perp | 250 | 15m_observation_then_1h_confirmation | 38.87 | 11.62 | 27.25 | 1.49 | 18351 | 0.0136 |
| PENGU | short_perp | 250 | 15m_observation_then_1h_confirmation | 31.76 | 18.85 | 12.91 | 8.72 | 18772 | 0.0133 |
| FET | short_perp | 250 | 15m_observation_then_1h_confirmation | 26.44 | 15.81 | 10.63 | 5.67 | 17357 | 0.0144 |
| SUI | short_perp | 250 | 15m_observation_then_1h_confirmation | 16.55 | 11.36 | 5.19 | 1.32 | 71564 | 0.0035 |

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
