# Current Microstructure Flow Paper Gate

This subtracts taker round-trip fees, current spread, and a rough visible-depth impact from microstructure flow labels. It is a small paper-probe gate, not a maker queue or fill model.

| asset | action | dir | size USD | gross15 bps | net15 bps | net1h bps | spread bps | depth USD | usage | gate | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| MEGA | aligned_pressure_watch | 1 | 100 | 185.80 | 170.42 | 21.65 | 5.17 | 4553 | 0.0220 | microstructure_small_paper_probe | survives rough fee, spread, 1h label, and visible-depth check |
| MEGA | aligned_pressure_watch | 1 | 250 | 185.80 | 170.09 | 21.32 | 5.17 | 4553 | 0.0549 | microstructure_small_paper_probe | survives rough fee, spread, 1h label, and visible-depth check |
| MEGA | aligned_pressure_watch | 1 | 500 | 185.80 | 169.54 | 20.77 | 5.17 | 4553 | 0.1098 | microstructure_small_paper_probe | survives rough fee, spread, 1h label, and visible-depth check |
| MEGA | aligned_pressure_watch | 1 | 1000 | 185.80 | 168.44 | 19.68 | 5.17 | 4553 | 0.2196 | microstructure_small_paper_probe | survives rough fee, spread, 1h label, and visible-depth check |
| HYPE | book_trade_divergence_watch | 1 | 100 | 158.35 | 148.18 | 261.34 | 0.16 | 186307 | 0.0005 | microstructure_small_paper_probe | survives rough fee, spread, 1h label, and visible-depth check |
| HYPE | book_trade_divergence_watch | 1 | 250 | 158.35 | 148.17 | 261.33 | 0.16 | 186307 | 0.0013 | microstructure_small_paper_probe | survives rough fee, spread, 1h label, and visible-depth check |
| HYPE | book_trade_divergence_watch | 1 | 500 | 158.35 | 148.16 | 261.32 | 0.16 | 186307 | 0.0027 | microstructure_small_paper_probe | survives rough fee, spread, 1h label, and visible-depth check |
| HYPE | book_trade_divergence_watch | 1 | 1000 | 158.35 | 148.13 | 261.29 | 0.16 | 186307 | 0.0054 | microstructure_small_paper_probe | survives rough fee, spread, 1h label, and visible-depth check |
| NEAR | aligned_pressure_watch | 1 | 100 | 121.89 | 104.50 | 363.21 | 7.31 | 12445 | 0.0080 | microstructure_small_paper_probe | survives rough fee, spread, 1h label, and visible-depth check |
| NEAR | aligned_pressure_watch | 1 | 250 | 121.89 | 104.38 | 363.09 | 7.31 | 12445 | 0.0201 | microstructure_small_paper_probe | survives rough fee, spread, 1h label, and visible-depth check |
| NEAR | aligned_pressure_watch | 1 | 500 | 121.89 | 104.18 | 362.89 | 7.31 | 12445 | 0.0402 | microstructure_small_paper_probe | survives rough fee, spread, 1h label, and visible-depth check |
| NEAR | aligned_pressure_watch | 1 | 1000 | 121.89 | 103.78 | 362.49 | 7.31 | 12445 | 0.0804 | microstructure_small_paper_probe | survives rough fee, spread, 1h label, and visible-depth check |
| SUI | aligned_pressure_watch | 1 | 100 | 110.85 | 100.70 | 120.56 | 0.14 | 75798 | 0.0013 | microstructure_small_paper_probe | survives rough fee, spread, 1h label, and visible-depth check |
| SUI | aligned_pressure_watch | 1 | 250 | 110.85 | 100.69 | 120.54 | 0.14 | 75798 | 0.0033 | microstructure_small_paper_probe | survives rough fee, spread, 1h label, and visible-depth check |
| SUI | aligned_pressure_watch | 1 | 500 | 110.85 | 100.65 | 120.51 | 0.14 | 75798 | 0.0066 | microstructure_small_paper_probe | survives rough fee, spread, 1h label, and visible-depth check |
| SUI | aligned_pressure_watch | 1 | 1000 | 110.85 | 100.59 | 120.44 | 0.14 | 75798 | 0.0132 | microstructure_small_paper_probe | survives rough fee, spread, 1h label, and visible-depth check |
| MON | aligned_pressure_watch | 1 | 100 | 88.51 | 73.26 | 74.65 | 4.64 | 1655 | 0.0604 | microstructure_small_paper_probe | survives rough fee, spread, 1h label, and visible-depth check |
| MON | aligned_pressure_watch | 1 | 250 | 88.51 | 72.35 | 73.74 | 4.64 | 1655 | 0.1511 | microstructure_small_paper_probe | survives rough fee, spread, 1h label, and visible-depth check |
| CHIP | book_trade_divergence_watch | 1 | 100 | 83.62 | 67.77 | 136.90 | 5.60 | 4059 | 0.0246 | microstructure_small_paper_probe | survives rough fee, spread, 1h label, and visible-depth check |
| CHIP | book_trade_divergence_watch | 1 | 250 | 83.62 | 67.40 | 136.53 | 5.60 | 4059 | 0.0616 | microstructure_small_paper_probe | survives rough fee, spread, 1h label, and visible-depth check |

## Interpretation

`microstructure_small_paper_probe` means the 15m and 1h directional labels survived a rough taker-fee, spread, and visible-depth check. It still needs real fill logs, queue/adverse-selection measurement, and repeat snapshots before it can be treated as a trading edge.
