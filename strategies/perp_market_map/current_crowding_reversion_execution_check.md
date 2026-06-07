# Current Crowding Reversion Execution Check

This applies a rough taker-fee, spread, and visible-depth gate to validated Hyperliquid carry-reversion candidates. It is still not a fill model.

- rows: `30`
- paper execution probes: `10`

| asset | action | size | gate | net 1h bps | cost bps | conservative bps | spread | depth10 | usage | reason |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| DYDX | short_carry_reversion_watch | 250 | paper_execution_probe | 358.79 | 19.23 | 339.56 | 8.31 | 2713 | 0.0921 | public book does not obviously block a small paper probe |
| ZRO | short_carry_reversion_watch | 250 | paper_execution_probe | 282.98 | 10.95 | 272.02 | 0.77 | 13754 | 0.0182 | public book does not obviously block a small paper probe |
| ZRO | short_carry_reversion_watch | 1000 | paper_execution_probe | 282.98 | 11.50 | 271.48 | 0.77 | 13754 | 0.0727 | public book does not obviously block a small paper probe |
| ZRO | short_carry_reversion_watch | 2500 | paper_execution_probe | 282.98 | 12.59 | 270.39 | 0.77 | 13754 | 0.1818 | public book does not obviously block a small paper probe |
| XMR | short_carry_reversion_watch | 250 | paper_execution_probe | 244.79 | 14.41 | 230.38 | 4.26 | 16896 | 0.0148 | public book does not obviously block a small paper probe |
| XMR | short_carry_reversion_watch | 1000 | paper_execution_probe | 244.79 | 14.85 | 229.94 | 4.26 | 16896 | 0.0592 | public book does not obviously block a small paper probe |
| XMR | short_carry_reversion_watch | 2500 | paper_execution_probe | 244.79 | 15.74 | 229.05 | 4.26 | 16896 | 0.1480 | public book does not obviously block a small paper probe |
| ETHFI | short_carry_reversion_watch | 250 | paper_execution_probe | 245.82 | 18.67 | 227.15 | 7.64 | 2407 | 0.1039 | public book does not obviously block a small paper probe |
| GRIFFAIN | short_carry_reversion_watch | 250 | paper_execution_probe | 243.96 | 18.44 | 225.52 | 7.28 | 2151 | 0.1162 | public book does not obviously block a small paper probe |
| APEX | short_carry_reversion_watch | 250 | paper_execution_probe | 245.75 | 22.78 | 222.96 | 11.47 | 1895 | 0.1319 | public book does not obviously block a small paper probe |
| CFX | short_carry_reversion_watch | 250 | wide_spread_watch | 237.51 | 31.04 | 206.47 | 18.71 | 1073 | 0.2331 | edge survives rough cost but spread is wide |
| DYDX | short_carry_reversion_watch | 1000 | too_large_for_visible_depth | 358.79 | 22.00 | 336.80 | 8.31 | 2713 | 0.3686 | candidate size uses too much visible near-touch depth |
| DYDX | short_carry_reversion_watch | 2500 | too_large_for_visible_depth | 358.79 | 27.52 | 331.27 | 8.31 | 2713 | 0.9215 | candidate size uses too much visible near-touch depth |
| ETHFI | short_carry_reversion_watch | 1000 | too_large_for_visible_depth | 245.82 | 21.79 | 224.03 | 7.64 | 2407 | 0.4155 | candidate size uses too much visible near-touch depth |
| GRIFFAIN | short_carry_reversion_watch | 1000 | too_large_for_visible_depth | 243.96 | 21.93 | 222.03 | 7.28 | 2151 | 0.4650 | candidate size uses too much visible near-touch depth |
| APEX | short_carry_reversion_watch | 1000 | too_large_for_visible_depth | 245.75 | 26.74 | 219.00 | 11.47 | 1895 | 0.5277 | candidate size uses too much visible near-touch depth |
| ETHFI | short_carry_reversion_watch | 2500 | too_large_for_visible_depth | 245.82 | 27.64 | 218.19 | 7.64 | 2407 | 1.0387 | candidate size uses too much visible near-touch depth |
| GRIFFAIN | short_carry_reversion_watch | 2500 | too_large_for_visible_depth | 243.96 | 27.28 | 216.68 | 7.28 | 2151 | 1.1624 | candidate size uses too much visible near-touch depth |
| APEX | short_carry_reversion_watch | 2500 | too_large_for_visible_depth | 245.75 | 31.47 | 214.28 | 11.47 | 1895 | 1.3193 | candidate size uses too much visible near-touch depth |
| HEMI | short_carry_reversion_watch | 250 | too_large_for_visible_depth | 228.20 | 22.34 | 205.87 | 8.97 | 742 | 0.3370 | candidate size uses too much visible near-touch depth |
| CFX | short_carry_reversion_watch | 1000 | too_large_for_visible_depth | 237.51 | 38.03 | 199.48 | 18.71 | 1073 | 0.9324 | candidate size uses too much visible near-touch depth |
| HEMI | short_carry_reversion_watch | 1000 | too_large_for_visible_depth | 228.20 | 28.97 | 199.24 | 8.97 | 742 | 1.3479 | candidate size uses too much visible near-touch depth |
| HEMI | short_carry_reversion_watch | 2500 | too_large_for_visible_depth | 228.20 | 28.97 | 199.24 | 8.97 | 742 | 3.3697 | candidate size uses too much visible near-touch depth |
| CFX | short_carry_reversion_watch | 2500 | too_large_for_visible_depth | 237.51 | 38.71 | 198.80 | 18.71 | 1073 | 2.3310 | candidate size uses too much visible near-touch depth |
| PURR | short_carry_reversion_watch | 250 | no_visible_depth | 95.93 | 51.81 | 44.13 | 31.81 | 0 | inf | no visible near-touch depth on the execution side |
| PURR | short_carry_reversion_watch | 1000 | no_visible_depth | 95.93 | 51.81 | 44.13 | 31.81 | 0 | inf | no visible near-touch depth on the execution side |
| PURR | short_carry_reversion_watch | 2500 | no_visible_depth | 95.93 | 51.81 | 44.13 | 31.81 | 0 | inf | no visible near-touch depth on the execution side |
| HMSTR | short_carry_reversion_watch | 250 | no_visible_depth | -54.52 | 73.62 | -128.14 | 53.62 | 0 | inf | no visible near-touch depth on the execution side |
| HMSTR | short_carry_reversion_watch | 1000 | no_visible_depth | -54.52 | 73.62 | -128.14 | 53.62 | 0 | inf | no visible near-touch depth on the execution side |
| HMSTR | short_carry_reversion_watch | 2500 | no_visible_depth | -54.52 | 73.62 | -128.14 | 53.62 | 0 | inf | no visible near-touch depth on the execution side |

## Interpretation

`paper_execution_probe` only means the current public book does not obviously block a small paper probe. It still excludes queue position, mark/index basis, stop behavior, funding timing, and repeated adverse selection.
