# Current Crowding Reversion Execution Check

This applies a rough taker-fee, spread, and visible-depth gate to validated Hyperliquid carry-reversion candidates. It is still not a fill model.

- rows: `0`
- paper execution probes: `0`

| asset | action | size | gate | net 1h bps | cost bps | conservative bps | spread | depth10 | usage | reason |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |

## Interpretation

`paper_execution_probe` only means the current public book does not obviously block a small paper probe. It still excludes queue position, mark/index basis, stop behavior, funding timing, and repeated adverse selection.
