# Current L2 Imbalance Paper Gate

This subtracts taker round-trip fees and current spread from the book-imbalance directional label, then checks visible 10 bps depth. It is a directional paper gate, not a maker-fill model.

| asset | size USD | imbalance10 | cost bps | net15 bps | net1h bps | depth USD | depth usage | gate | reason |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| HYPE | 100 | 0.2310 | 10.17 | 122.57 | 199.97 | 60570 | 0.0017 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| HYPE | 250 | 0.2310 | 10.17 | 122.57 | 199.97 | 60570 | 0.0041 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| HYPE | 500 | 0.2310 | 10.17 | 122.57 | 199.97 | 60570 | 0.0083 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| HYPE | 1000 | 0.2310 | 10.17 | 122.57 | 199.97 | 60570 | 0.0165 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| HYPE | 2500 | 0.2310 | 10.17 | 122.57 | 199.97 | 60570 | 0.0413 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| HYPE | 5000 | 0.2310 | 10.17 | 122.57 | 199.97 | 60570 | 0.0825 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| SOL | 100 | 0.4100 | 10.16 | 79.64 | 119.41 | 319875 | 0.0003 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| SOL | 250 | 0.4100 | 10.16 | 79.64 | 119.41 | 319875 | 0.0008 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| SOL | 500 | 0.4100 | 10.16 | 79.64 | 119.41 | 319875 | 0.0016 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| SOL | 1000 | 0.4100 | 10.16 | 79.64 | 119.41 | 319875 | 0.0031 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| SOL | 2500 | 0.4100 | 10.16 | 79.64 | 119.41 | 319875 | 0.0078 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| SOL | 5000 | 0.4100 | 10.16 | 79.64 | 119.41 | 319875 | 0.0156 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| BTC | 100 | 0.0167 | 10.16 | 19.41 | 18.93 | 3670441 | 0.0000 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| BTC | 250 | 0.0167 | 10.16 | 19.41 | 18.93 | 3670441 | 0.0001 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| BTC | 500 | 0.0167 | 10.16 | 19.41 | 18.93 | 3670441 | 0.0001 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| BTC | 1000 | 0.0167 | 10.16 | 19.41 | 18.93 | 3670441 | 0.0003 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| BTC | 2500 | 0.0167 | 10.16 | 19.41 | 18.93 | 3670441 | 0.0007 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| BTC | 5000 | 0.0167 | 10.16 | 19.41 | 18.93 | 3670441 | 0.0014 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| ETH | 100 | -0.0338 | 10.63 | -68.00 | -115.92 | 10955176 | 0.0000 | blocked_by_cost | fee and spread consume the 15m directional label |
| ETH | 250 | -0.0338 | 10.63 | -68.00 | -115.92 | 10955176 | 0.0000 | blocked_by_cost | fee and spread consume the 15m directional label |

## Interpretation

`small_paper_probe` means the imbalance direction survived the rough fee/spread/depth check at that notional. This does not prove a market making edge because queue position, fill probability, rebates, and adverse selection are still unmodeled.
