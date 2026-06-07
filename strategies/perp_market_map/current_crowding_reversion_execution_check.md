# Current Crowding Reversion Execution Check

This applies a rough taker-fee, spread, and visible-depth gate to validated Hyperliquid carry-reversion candidates. It is still not a fill model.

- rows: `30`
- paper execution probes: `11`

| asset | action | size | gate | net 1h bps | cost bps | conservative bps | spread | depth10 | usage | reason |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| DYDX | short_carry_reversion_watch | 250 | paper_execution_probe | 358.79 | 17.81 | 340.98 | 6.92 | 2783 | 0.0898 | public book does not obviously block a small paper probe |
| ZRO | short_carry_reversion_watch | 250 | paper_execution_probe | 282.98 | 18.30 | 264.68 | 7.99 | 8143 | 0.0307 | public book does not obviously block a small paper probe |
| ZRO | short_carry_reversion_watch | 1000 | paper_execution_probe | 282.98 | 19.22 | 263.76 | 7.99 | 8143 | 0.1228 | public book does not obviously block a small paper probe |
| ETHFI | short_carry_reversion_watch | 250 | paper_execution_probe | 245.82 | 11.93 | 233.89 | 1.33 | 4154 | 0.0602 | public book does not obviously block a small paper probe |
| XMR | short_carry_reversion_watch | 250 | paper_execution_probe | 244.79 | 12.17 | 232.62 | 1.97 | 12843 | 0.0195 | public book does not obviously block a small paper probe |
| ETHFI | short_carry_reversion_watch | 1000 | paper_execution_probe | 245.82 | 13.74 | 232.08 | 1.33 | 4154 | 0.2407 | public book does not obviously block a small paper probe |
| XMR | short_carry_reversion_watch | 1000 | paper_execution_probe | 244.79 | 12.75 | 232.04 | 1.97 | 12843 | 0.0779 | public book does not obviously block a small paper probe |
| XMR | short_carry_reversion_watch | 2500 | paper_execution_probe | 244.79 | 13.92 | 230.87 | 1.97 | 12843 | 0.1947 | public book does not obviously block a small paper probe |
| CFX | short_carry_reversion_watch | 250 | paper_execution_probe | 237.51 | 14.53 | 222.98 | 3.93 | 4181 | 0.0598 | public book does not obviously block a small paper probe |
| CFX | short_carry_reversion_watch | 1000 | paper_execution_probe | 237.51 | 16.32 | 221.19 | 3.93 | 4181 | 0.2392 | public book does not obviously block a small paper probe |
| HEMI | short_carry_reversion_watch | 250 | paper_execution_probe | 228.20 | 18.38 | 209.82 | 7.18 | 2082 | 0.1201 | public book does not obviously block a small paper probe |
| DYDX | short_carry_reversion_watch | 1000 | too_large_for_visible_depth | 358.79 | 20.51 | 338.28 | 6.92 | 2783 | 0.3593 | candidate size uses too much visible near-touch depth |
| DYDX | short_carry_reversion_watch | 2500 | too_large_for_visible_depth | 358.79 | 25.90 | 332.90 | 6.92 | 2783 | 0.8982 | candidate size uses too much visible near-touch depth |
| ZRO | short_carry_reversion_watch | 2500 | too_large_for_visible_depth | 282.98 | 21.06 | 261.92 | 7.99 | 8143 | 0.3070 | candidate size uses too much visible near-touch depth |
| ETHFI | short_carry_reversion_watch | 2500 | too_large_for_visible_depth | 245.82 | 17.35 | 228.47 | 1.33 | 4154 | 0.6018 | candidate size uses too much visible near-touch depth |
| APEX | short_carry_reversion_watch | 250 | too_large_for_visible_depth | 245.75 | 23.32 | 222.43 | 9.57 | 666 | 0.3753 | candidate size uses too much visible near-touch depth |
| CFX | short_carry_reversion_watch | 2500 | too_large_for_visible_depth | 237.51 | 19.91 | 217.60 | 3.93 | 4181 | 0.5979 | candidate size uses too much visible near-touch depth |
| APEX | short_carry_reversion_watch | 1000 | too_large_for_visible_depth | 245.75 | 29.57 | 216.18 | 9.57 | 666 | 1.5013 | candidate size uses too much visible near-touch depth |
| APEX | short_carry_reversion_watch | 2500 | too_large_for_visible_depth | 245.75 | 29.57 | 216.18 | 9.57 | 666 | 3.7531 | candidate size uses too much visible near-touch depth |
| GRIFFAIN | short_carry_reversion_watch | 250 | too_large_for_visible_depth | 243.96 | 35.54 | 208.43 | 19.50 | 414 | 0.6035 | candidate size uses too much visible near-touch depth |
| HEMI | short_carry_reversion_watch | 1000 | too_large_for_visible_depth | 228.20 | 21.98 | 206.22 | 7.18 | 2082 | 0.4803 | candidate size uses too much visible near-touch depth |
| GRIFFAIN | short_carry_reversion_watch | 1000 | too_large_for_visible_depth | 243.96 | 39.50 | 204.46 | 19.50 | 414 | 2.4141 | candidate size uses too much visible near-touch depth |
| GRIFFAIN | short_carry_reversion_watch | 2500 | too_large_for_visible_depth | 243.96 | 39.50 | 204.46 | 19.50 | 414 | 6.0353 | candidate size uses too much visible near-touch depth |
| HEMI | short_carry_reversion_watch | 2500 | too_large_for_visible_depth | 228.20 | 27.18 | 201.02 | 7.18 | 2082 | 1.2008 | candidate size uses too much visible near-touch depth |
| PURR | short_carry_reversion_watch | 250 | no_visible_depth | 95.93 | 44.91 | 51.02 | 24.91 | 0 | inf | no visible near-touch depth on the execution side |
| PURR | short_carry_reversion_watch | 1000 | no_visible_depth | 95.93 | 44.91 | 51.02 | 24.91 | 0 | inf | no visible near-touch depth on the execution side |
| PURR | short_carry_reversion_watch | 2500 | no_visible_depth | 95.93 | 44.91 | 51.02 | 24.91 | 0 | inf | no visible near-touch depth on the execution side |
| HMSTR | short_carry_reversion_watch | 250 | no_visible_depth | -54.52 | 73.33 | -127.85 | 53.33 | 0 | inf | no visible near-touch depth on the execution side |
| HMSTR | short_carry_reversion_watch | 1000 | no_visible_depth | -54.52 | 73.33 | -127.85 | 53.33 | 0 | inf | no visible near-touch depth on the execution side |
| HMSTR | short_carry_reversion_watch | 2500 | no_visible_depth | -54.52 | 73.33 | -127.85 | 53.33 | 0 | inf | no visible near-touch depth on the execution side |

## Interpretation

`paper_execution_probe` only means the current public book does not obviously block a small paper probe. It still excludes queue position, mark/index basis, stop behavior, funding timing, and repeated adverse selection.
