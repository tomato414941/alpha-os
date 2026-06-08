# Current Hyperliquid Dislocation Paper Tickets

This is not a trade instruction. It converts current 15m-supported dislocation probes into paper observation and falsification tickets.

- generated tickets: `2`

| asset | side | notional | horizon | gross15 | cost | net15 | spread | depth10 | usage |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ZEC | short_perp | 250 | 15m_observation_then_1h_confirmation | 40.01 | 12.60 | 27.41 | 2.56 | 67915 | 0.0037 |
| MANTA | short_perp | 250 | 15m_observation_then_1h_confirmation | 33.78 | 16.02 | 17.77 | 5.27 | 3332 | 0.0750 |

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
