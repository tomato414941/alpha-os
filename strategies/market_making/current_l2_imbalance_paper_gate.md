# Current L2 Imbalance Paper Gate

This subtracts taker round-trip fees and current spread from the book-imbalance directional label, then checks visible 10 bps depth. It is a directional paper gate, not a maker-fill model.

| asset | size USD | imbalance10 | cost bps | net15 bps | net1h bps | depth USD | depth usage | gate | reason |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ETH | 100 | 0.1378 | 10.59 | 61.78 | -27.80 | 9998689 | 0.0000 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| ETH | 250 | 0.1378 | 10.59 | 61.78 | -27.80 | 9998689 | 0.0000 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| ETH | 500 | 0.1378 | 10.59 | 61.78 | -27.80 | 9998689 | 0.0001 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| ETH | 1000 | 0.1378 | 10.59 | 61.78 | -27.80 | 9998689 | 0.0001 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| ETH | 2500 | 0.1378 | 10.59 | 61.78 | -27.80 | 9998689 | 0.0003 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| ETH | 5000 | 0.1378 | 10.59 | 61.78 | -27.80 | 9998689 | 0.0005 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| SUI | 100 | 0.3281 | 11.98 | 36.92 | -34.84 | 46780 | 0.0021 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| SUI | 250 | 0.3281 | 11.98 | 36.92 | -34.84 | 46780 | 0.0053 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| SUI | 500 | 0.3281 | 11.98 | 36.92 | -34.84 | 46780 | 0.0107 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| SUI | 1000 | 0.3281 | 11.98 | 36.92 | -34.84 | 46780 | 0.0214 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| SUI | 2500 | 0.3281 | 11.98 | 36.92 | -34.84 | 46780 | 0.0534 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| SUI | 5000 | 0.3281 | 11.98 | 36.92 | -34.84 | 46780 | 0.1069 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| BTC | 100 | 0.7403 | 10.16 | 33.03 | -36.57 | 1341285 | 0.0001 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| BTC | 250 | 0.7403 | 10.16 | 33.03 | -36.57 | 1341285 | 0.0002 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| BTC | 500 | 0.7403 | 10.16 | 33.03 | -36.57 | 1341285 | 0.0004 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| BTC | 1000 | 0.7403 | 10.16 | 33.03 | -36.57 | 1341285 | 0.0007 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| BTC | 2500 | 0.7403 | 10.16 | 33.03 | -36.57 | 1341285 | 0.0019 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| BTC | 5000 | 0.7403 | 10.16 | 33.03 | -36.57 | 1341285 | 0.0037 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| DOGE | 100 | 0.0843 | 11.05 | 28.18 | -74.83 | 173677 | 0.0006 | small_paper_probe | survives rough fee, spread, and visible-depth check |
| DOGE | 250 | 0.0843 | 11.05 | 28.18 | -74.83 | 173677 | 0.0014 | small_paper_probe | survives rough fee, spread, and visible-depth check |

## Interpretation

`small_paper_probe` means the imbalance direction survived the rough fee/spread/depth check at that notional. This does not prove a market making edge because queue position, fill probability, rebates, and adverse selection are still unmodeled.
