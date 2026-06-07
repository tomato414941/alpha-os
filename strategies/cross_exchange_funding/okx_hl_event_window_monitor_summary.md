# OKX-Hyperliquid Event Window Monitor

This repeats event-window triage to check whether the current candidate classification is stable. It is not a trade instruction.

| asset | obs | dominant action | paper 8h rate | active 24h rate | watch rate | drop rate | mean very-low 8h | mean low-fee 24h | mean one-bps 24h | capacity |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 6 | paper_8h_candidate | 1.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.0000114 | 0.00010974 | -0.00009026 | 422448.80855333 |
| ZEC | 6 | fee_dependent_24h_monitor | 0.00000000 | 0.00000000 | 1.00000000 | 0.00000000 | -0.00041023 | 0.00006702 | -0.00013298 | 106210.05564167 |
| BABY | 6 | drop_for_now | 0.00000000 | 0.00000000 | 0.00000000 | 1.00000000 | -0.00187223 | -0.0003967 | -0.0005967 | 18182.63949194 |
| JTO | 6 | drop_for_now | 0.00000000 | 0.00000000 | 0.00000000 | 1.00000000 | -0.00149631 | -0.00143752 | -0.00163752 | 54543.56750198 |

## Interpretation

A candidate should not move toward paper execution unless the event-window action is stable and the surviving scenario is realistic for the account's actual fee and maker-fill conditions.
