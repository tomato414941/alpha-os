# Current Hyperliquid Dislocation Repeat Label Queue

This queue turns repeated monitor observations into the next labeling or paper-probe actions. It is a workflow queue, not a strategy or trade instruction.

| asset | status | side | action | priority | obs | mean score | net15 | out15 | gate | net15 gated | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |
| JTO | paper_crowded_momentum_continuation_candidate | long_perp | fresh_forward_label_candidate | 44.1005 | 17 | 32.1005 |  |  |  |  | rerun JTO forward label on a fresh repeated monitor window |
| STABLE | paper_extreme_funding_carry_candidate | long_perp | fresh_forward_label_candidate | 43.5808 | 17 | 31.5808 |  |  |  |  | rerun STABLE forward label on a fresh repeated monitor window |
| ZEC | paper_crowded_momentum_continuation_candidate | long_perp | fresh_forward_label_candidate | 42.0177 | 17 | 30.0177 |  |  |  |  | rerun ZEC forward label on a fresh repeated monitor window |
| ZEC | paper_crowded_momentum_reversal_candidate | short_perp | repeat_paper_probe_candidate | 40.6386 | 17 | 25.5151 |  |  | paper_execution_probe | 31.23533694 | repeat ZEC paper probe on a fresh snapshot and record fill/outcome evidence |
| JTO | paper_crowded_momentum_reversal_candidate | short_perp | fresh_forward_label_candidate | 39.2854 | 17 | 27.2854 |  |  |  |  | rerun JTO forward label on a fresh repeated monitor window |
| LDO | paper_crowded_momentum_continuation_candidate | long_perp | repeat_forward_label_priority | 34.7503 | 17 | 17.4856 | 52.64723806 | paper_15m_win |  |  | wait for LDO 1h label, then rerun execution check if still visible |
| STBL | paper_crowded_momentum_continuation_candidate | long_perp | fresh_forward_label_candidate | 33.3324 | 17 | 21.3324 |  |  |  |  | rerun STBL forward label on a fresh repeated monitor window |
| MON | paper_crowded_momentum_continuation_candidate | long_perp | repeat_forward_label_priority | 32.9450 | 13 | 16.2912 | 46.53800191 | paper_15m_win |  |  | wait for MON 1h label, then rerun execution check if still visible |
| DASH | paper_crowded_momentum_continuation_candidate | long_perp | repeat_forward_label_priority | 32.5493 | 17 | 20.2871 | 2.62242850 | paper_15m_win |  |  | wait for DASH 1h label, then rerun execution check if still visible |
| EIGEN | paper_crowded_momentum_continuation_candidate | long_perp | fresh_forward_label_candidate | 32.3007 | 17 | 20.3007 |  |  |  |  | rerun EIGEN forward label on a fresh repeated monitor window |
| WLD | paper_crowded_momentum_continuation_candidate | long_perp | fresh_forward_label_candidate | 31.8494 | 17 | 19.8494 |  |  |  |  | rerun WLD forward label on a fresh repeated monitor window |
| WLD | paper_crowded_momentum_reversal_candidate | short_perp | repeat_paper_probe_candidate | 31.5968 | 17 | 16.8720 |  |  | paper_execution_probe | 27.24823114 | repeat WLD paper probe on a fresh snapshot and record fill/outcome evidence |
| MANTA | paper_crowded_momentum_continuation_candidate | short_perp | fresh_forward_label_candidate | 30.8764 | 17 | 18.8764 |  |  |  |  | rerun MANTA forward label on a fresh repeated monitor window |
| TAO | paper_crowded_momentum_continuation_candidate | long_perp | fresh_forward_label_candidate | 30.7459 | 17 | 18.7459 |  |  |  |  | rerun TAO forward label on a fresh repeated monitor window |
| LINK | paper_crowded_momentum_continuation_candidate | long_perp | fresh_forward_label_candidate | 30.5457 | 17 | 18.5457 |  |  |  |  | rerun LINK forward label on a fresh repeated monitor window |
| STBL | paper_crowded_momentum_reversal_candidate | short_perp | fresh_forward_label_candidate | 30.1325 | 17 | 18.1325 |  |  |  |  | rerun STBL forward label on a fresh repeated monitor window |
| AERO | paper_crowded_momentum_continuation_candidate | long_perp | fresh_forward_label_candidate | 29.8187 | 17 | 17.8187 |  |  |  |  | rerun AERO forward label on a fresh repeated monitor window |
| MEGA | paper_crowded_momentum_continuation_candidate | long_perp | fresh_forward_label_candidate | 29.4007 | 17 | 17.4007 |  |  |  |  | rerun MEGA forward label on a fresh repeated monitor window |
| NEAR | paper_crowded_momentum_continuation_candidate | long_perp | fresh_forward_label_candidate | 29.3133 | 17 | 17.3133 |  |  |  |  | rerun NEAR forward label on a fresh repeated monitor window |
| EIGEN | paper_crowded_momentum_reversal_candidate | short_perp | fresh_forward_label_candidate | 29.2556 | 17 | 17.2556 |  |  |  |  | rerun EIGEN forward label on a fresh repeated monitor window |
| DASH | paper_crowded_momentum_reversal_candidate | short_perp | monitor_conflict_relabel | 29.2440 | 17 | 17.2440 | -48.10254752 | paper_15m_loss |  |  | wait for DASH 1h label, then rerun execution check if still visible |
| PURR | paper_crowded_momentum_continuation_candidate | long_perp | monitor_conflict_relabel | 29.1202 | 17 | 17.1202 | -94.52272495 | paper_15m_loss |  |  | wait for PURR 1h label, then rerun execution check if still visible |
| PENGU | paper_crowded_momentum_continuation_candidate | long_perp | fresh_forward_label_candidate | 28.8621 | 17 | 16.8621 |  |  |  |  | rerun PENGU forward label on a fresh repeated monitor window |
| FARTCOIN | paper_crowded_momentum_continuation_candidate | long_perp | fresh_forward_label_candidate | 28.6166 | 17 | 16.6166 |  |  |  |  | rerun FARTCOIN forward label on a fresh repeated monitor window |
| DYDX | paper_crowded_momentum_continuation_candidate | long_perp | fresh_forward_label_candidate | 28.5369 | 17 | 16.5369 |  |  |  |  | rerun DYDX forward label on a fresh repeated monitor window |
| PUMP | paper_crowded_momentum_continuation_candidate | long_perp | fresh_forward_label_candidate | 28.2055 | 17 | 16.2055 |  |  |  |  | rerun PUMP forward label on a fresh repeated monitor window |
| MANTA | paper_crowded_momentum_reversal_candidate | long_perp | fresh_forward_label_candidate | 28.0449 | 17 | 16.0449 |  |  |  |  | rerun MANTA forward label on a fresh repeated monitor window |
| PENGU | paper_crowded_momentum_reversal_candidate | short_perp | repeat_paper_probe_candidate | 27.9424 | 13 | 14.6512 |  |  | paper_execution_probe | 12.91203027 | repeat PENGU paper probe on a fresh snapshot and record fill/outcome evidence |
| TAO | paper_crowded_momentum_reversal_candidate | short_perp | fresh_forward_label_candidate | 27.9340 | 17 | 15.9340 |  |  |  |  | rerun TAO forward label on a fresh repeated monitor window |
| LINK | paper_crowded_momentum_reversal_candidate | short_perp | fresh_forward_label_candidate | 27.7639 | 17 | 15.7639 |  |  |  |  | rerun LINK forward label on a fresh repeated monitor window |
