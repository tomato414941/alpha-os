# Current L2 Imbalance Paper Gate

This subtracts taker round-trip fees and current spread from the book-imbalance directional label, then checks visible 10 bps depth. It is a directional paper gate, not a maker-fill model.

| asset | size USD | imbalance10 | cost bps | net15 bps | net1h bps | depth USD | depth usage | gate | reason |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| JTO | 100 | 0.2901 | 14.68 | 110.06 |  | 4160 | 0.0240 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| JTO | 250 | 0.2901 | 14.68 | 110.06 |  | 4160 | 0.0601 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| JTO | 500 | 0.2901 | 14.68 | 110.06 |  | 4160 | 0.1202 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| JTO | 1000 | 0.2901 | 14.68 | 110.06 |  | 4160 | 0.2404 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| XLM | 100 | 0.1395 | 12.48 | 109.39 |  | 21244 | 0.0047 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| XLM | 250 | 0.1395 | 12.48 | 109.39 |  | 21244 | 0.0118 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| XLM | 500 | 0.1395 | 12.48 | 109.39 |  | 21244 | 0.0235 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| XLM | 1000 | 0.1395 | 12.48 | 109.39 |  | 21244 | 0.0471 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| XLM | 2500 | 0.1395 | 12.48 | 109.39 |  | 21244 | 0.1177 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| XLM | 5000 | 0.1395 | 12.48 | 109.39 |  | 21244 | 0.2354 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| NEAR | 100 | -0.2800 | 11.92 | 103.92 |  | 21602 | 0.0046 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| NEAR | 250 | -0.2800 | 11.92 | 103.92 |  | 21602 | 0.0116 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| NEAR | 500 | -0.2800 | 11.92 | 103.92 |  | 21602 | 0.0231 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| NEAR | 1000 | -0.2800 | 11.92 | 103.92 |  | 21602 | 0.0463 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| NEAR | 2500 | -0.2800 | 11.92 | 103.92 |  | 21602 | 0.1157 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| NEAR | 5000 | -0.2800 | 11.92 | 103.92 |  | 21602 | 0.2315 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| XPL | 100 | 0.0082 | 12.18 | 17.96 |  | 10261 | 0.0097 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| XPL | 250 | 0.0082 | 12.18 | 17.96 |  | 10261 | 0.0244 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| XPL | 500 | 0.0082 | 12.18 | 17.96 |  | 10261 | 0.0487 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| XPL | 1000 | 0.0082 | 12.18 | 17.96 |  | 10261 | 0.0975 | small_paper_probe | survives rough fee, spread, and visible-depth check |

## Interpretation

`small_paper_probe` means the imbalance direction survived the rough fee/spread/depth check at that notional. This does not prove a market making edge because queue position, fill probability, rebates, and adverse selection are still unmodeled.
