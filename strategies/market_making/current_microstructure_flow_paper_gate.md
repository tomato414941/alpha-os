# Current Microstructure Flow Paper Gate

This subtracts taker round-trip fees, current spread, and a rough visible-depth impact from microstructure flow labels. It is a small paper-probe gate, not a maker queue or fill model.

| asset | action | dir | size USD | gross15 bps | net15 bps | net1h bps | spread bps | depth USD | usage | gate | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SUI | book_trade_divergence_watch | -1 | 100 | 0.00 | -10.14 | -10.14 | 0.13 | 79609 | 0.0013 | wait_for_1h_label | 1h label is not mature yet |
| BTC | aligned_pressure_watch | 1 | 100 | 0.00 | -10.16 | -10.16 | 0.16 | 6431380 | 0.0000 | wait_for_1h_label | 1h label is not mature yet |
| BTC | aligned_pressure_watch | 1 | 250 | 0.00 | -10.16 | -10.16 | 0.16 | 6431380 | 0.0000 | wait_for_1h_label | 1h label is not mature yet |
| BTC | aligned_pressure_watch | 1 | 500 | 0.00 | -10.16 | -10.16 | 0.16 | 6431380 | 0.0001 | wait_for_1h_label | 1h label is not mature yet |
| BTC | aligned_pressure_watch | 1 | 1000 | 0.00 | -10.16 | -10.16 | 0.16 | 6431380 | 0.0002 | wait_for_1h_label | 1h label is not mature yet |
| SUI | book_trade_divergence_watch | -1 | 250 | 0.00 | -10.16 | -10.16 | 0.13 | 79609 | 0.0031 | wait_for_1h_label | 1h label is not mature yet |
| HYPE | aligned_pressure_watch | 1 | 100 | 0.00 | -10.17 | -10.17 | 0.16 | 119267 | 0.0008 | wait_for_1h_label | 1h label is not mature yet |
| HYPE | aligned_pressure_watch | 1 | 250 | 0.00 | -10.18 | -10.18 | 0.16 | 119267 | 0.0021 | wait_for_1h_label | 1h label is not mature yet |
| SUI | book_trade_divergence_watch | -1 | 500 | 0.00 | -10.19 | -10.19 | 0.13 | 79609 | 0.0063 | wait_for_1h_label | 1h label is not mature yet |
| HYPE | aligned_pressure_watch | 1 | 500 | 0.00 | -10.20 | -10.20 | 0.16 | 119267 | 0.0042 | wait_for_1h_label | 1h label is not mature yet |
| HYPE | aligned_pressure_watch | 1 | 1000 | 0.00 | -10.25 | -10.25 | 0.16 | 119267 | 0.0084 | wait_for_1h_label | 1h label is not mature yet |
| SUI | book_trade_divergence_watch | -1 | 1000 | 0.00 | -10.26 | -10.26 | 0.13 | 79609 | 0.0126 | wait_for_1h_label | 1h label is not mature yet |
| NEAR | aligned_pressure_watch | 1 | 100 | 0.00 | -10.52 | -10.52 | 0.46 | 16700 | 0.0060 | wait_for_1h_label | 1h label is not mature yet |
| ETH | aligned_pressure_watch | 1 | 100 | 0.00 | -10.60 | -10.60 | 0.60 | 11207347 | 0.0000 | wait_for_1h_label | 1h label is not mature yet |
| ETH | aligned_pressure_watch | 1 | 250 | 0.00 | -10.60 | -10.60 | 0.60 | 11207347 | 0.0000 | wait_for_1h_label | 1h label is not mature yet |
| ETH | aligned_pressure_watch | 1 | 500 | 0.00 | -10.60 | -10.60 | 0.60 | 11207347 | 0.0000 | wait_for_1h_label | 1h label is not mature yet |
| ETH | aligned_pressure_watch | 1 | 1000 | 0.00 | -10.60 | -10.60 | 0.60 | 11207347 | 0.0001 | wait_for_1h_label | 1h label is not mature yet |
| SOL | aligned_pressure_watch | 1 | 100 | 0.00 | -10.60 | -10.60 | 0.60 | 563534 | 0.0002 | wait_for_1h_label | 1h label is not mature yet |
| SOL | aligned_pressure_watch | 1 | 250 | 0.00 | -10.61 | -10.61 | 0.60 | 563534 | 0.0004 | wait_for_1h_label | 1h label is not mature yet |
| SOL | aligned_pressure_watch | 1 | 500 | 0.00 | -10.61 | -10.61 | 0.60 | 563534 | 0.0009 | wait_for_1h_label | 1h label is not mature yet |

## Interpretation

`microstructure_small_paper_probe` means the 15m and 1h directional labels survived a rough taker-fee, spread, and visible-depth check. It still needs real fill logs, queue/adverse-selection measurement, and repeat snapshots before it can be treated as a trading edge.
