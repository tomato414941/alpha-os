# Current Hyperliquid Dislocation Forward Labels

This labels Hyperliquid dislocation candidates after rough taker fees, impact spread, and funding carry. It is still paper labeling, not a live fill or deployable strategy.

- rows: `151`
- covered 15m: `133`
- covered 1h: `0`
- covered 4h: `0`

| asset | status | side | score | cost bps | net15 | out15 | net1h | out1h | net4h | out4h |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- | ---: | --- |
| ZEC | paper_crowded_momentum_reversal_candidate | short_perp | 25.4635 | 16.75 | 25.67 | paper_15m_win |  | pending_1h |  | pending_4h |
| WLD | paper_crowded_momentum_reversal_candidate | short_perp | 20.2026 | 22.34 | 16.53 | paper_15m_win |  | pending_1h |  | pending_4h |
| PENGU | paper_crowded_momentum_reversal_candidate | short_perp | 15.0072 | 15.73 | 16.03 | paper_15m_win |  | pending_1h |  | pending_4h |
| FET | paper_crowded_momentum_reversal_candidate | short_perp | 11.3057 | 20.71 | 5.74 | paper_15m_win |  | pending_1h |  | pending_4h |
| SUI | paper_crowded_momentum_reversal_candidate | short_perp | 9.8298 | 14.35 | 2.20 | paper_15m_win |  | pending_1h |  | pending_4h |
| SOL | paper_crowded_momentum_reversal_candidate | short_perp | 11.2100 | 11.66 | -0.16 | paper_15m_loss |  | pending_1h |  | pending_4h |
| ETH | paper_crowded_momentum_reversal_candidate | short_perp | 12.6833 | 10.83 | -2.49 | paper_15m_loss |  | pending_1h |  | pending_4h |
| DOGE | paper_crowded_momentum_reversal_candidate | short_perp | 10.0791 | 12.78 | -5.30 | paper_15m_loss |  | pending_1h |  | pending_4h |
| BNB | paper_crowded_momentum_reversal_candidate | short_perp | 11.4155 | 11.49 | -6.49 | paper_15m_loss |  | pending_1h |  | pending_4h |
| VIRTUAL | paper_crowded_momentum_reversal_candidate | short_perp | 11.8675 | 19.32 | -8.10 | paper_15m_loss |  | pending_1h |  | pending_4h |
| TAO | paper_crowded_momentum_reversal_candidate | short_perp | 16.1050 | 20.47 | -8.70 | paper_15m_loss |  | pending_1h |  | pending_4h |
| XRP | paper_crowded_momentum_reversal_candidate | short_perp | 10.3947 | 12.55 | -9.06 | paper_15m_loss |  | pending_1h |  | pending_4h |
| LINK | paper_crowded_momentum_reversal_candidate | short_perp | 15.9666 | 15.33 | -9.73 | paper_15m_loss |  | pending_1h |  | pending_4h |
| HYPE | paper_crowded_momentum_reversal_candidate | short_perp | 11.3094 | 13.63 | -9.78 | paper_15m_loss |  | pending_1h |  | pending_4h |
| MEME | paper_crowded_momentum_continuation_candidate | long_perp | 11.8477 | 26.95 | -10.00 | paper_15m_loss |  | pending_1h |  | pending_4h |
| PUMP | paper_crowded_momentum_reversal_candidate | short_perp | 13.8941 | 23.11 | -10.04 | paper_15m_loss |  | pending_1h |  | pending_4h |
| ADA | paper_crowded_momentum_reversal_candidate | short_perp | 9.4061 | 12.91 | -10.45 | paper_15m_loss |  | pending_1h |  | pending_4h |
| BCH | paper_crowded_momentum_continuation_candidate | long_perp | 13.3209 | 17.56 | -11.49 | paper_15m_loss |  | pending_1h |  | pending_4h |
| NEAR | paper_crowded_momentum_reversal_candidate | short_perp | 15.0553 | 18.81 | -12.41 | paper_15m_loss |  | pending_1h |  | pending_4h |
| kPEPE | paper_crowded_momentum_reversal_candidate | short_perp | 10.1428 | 13.55 | -13.52 | paper_15m_loss |  | pending_1h |  | pending_4h |
| kPEPE | paper_crowded_momentum_continuation_candidate | long_perp | 11.9327 | 13.55 | -13.58 | paper_15m_loss |  | pending_1h |  | pending_4h |
| XPL | paper_crowded_momentum_continuation_candidate | long_perp | 12.2519 | 22.26 | -13.60 | paper_15m_loss |  | pending_1h |  | pending_4h |
| INJ | paper_crowded_momentum_continuation_candidate | long_perp | 11.7081 | 25.91 | -14.35 | paper_15m_loss |  | pending_1h |  | pending_4h |
| ADA | paper_crowded_momentum_continuation_candidate | long_perp | 11.0660 | 12.91 | -15.37 | paper_15m_loss |  | pending_1h |  | pending_4h |
| XMR | paper_extreme_funding_carry_candidate | short_perp | 11.9810 | 23.71 | -15.48 | paper_15m_loss |  | pending_1h |  | pending_4h |
| XMR | paper_mark_oracle_reversion_candidate | short_perp | 9.6139 | 23.71 | -15.48 | paper_15m_loss |  | pending_1h |  | pending_4h |
| TRUMP | paper_crowded_momentum_reversal_candidate | short_perp | 12.4271 | 17.45 | -15.81 | paper_15m_loss |  | pending_1h |  | pending_4h |
| SEI | paper_crowded_momentum_reversal_candidate | short_perp | 10.6212 | 20.10 | -15.86 | paper_15m_loss |  | pending_1h |  | pending_4h |
| STRK | paper_crowded_momentum_continuation_candidate | long_perp | 14.3422 | 21.86 | -15.94 | paper_15m_loss |  | pending_1h |  | pending_4h |
| XRP | paper_crowded_momentum_continuation_candidate | long_perp | 12.2290 | 12.55 | -16.04 | paper_15m_loss |  | pending_1h |  | pending_4h |

## Interpretation

Positive net means the candidate side beat rough taker costs plus the funding carry estimate over that horizon. Pending rows simply have not had enough elapsed time since the candidate snapshot.
