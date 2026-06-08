# Current Microstructure Flow Paper Gate

This subtracts taker round-trip fees, current spread, and a rough visible-depth impact from microstructure flow labels. It is a small paper-probe gate, not a maker queue or fill model.

| asset | action | dir | size USD | gross15 bps | net15 bps | net1h bps | spread bps | depth USD | usage | gate | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SOL | book_trade_divergence_watch | -1 | 100 | 0.00 | -10.15 | -10.15 | 0.15 | 199392 | 0.0005 | wait_for_1h_label | 1h label is not mature yet |
| BTC | aligned_pressure_watch | 1 | 100 | 0.00 | -10.16 | -10.16 | 0.16 | 3561930 | 0.0000 | wait_for_1h_label | 1h label is not mature yet |
| BTC | aligned_pressure_watch | 1 | 250 | 0.00 | -10.16 | -10.16 | 0.16 | 3561930 | 0.0001 | wait_for_1h_label | 1h label is not mature yet |
| BTC | aligned_pressure_watch | 1 | 500 | 0.00 | -10.16 | -10.16 | 0.16 | 3561930 | 0.0001 | wait_for_1h_label | 1h label is not mature yet |
| BTC | aligned_pressure_watch | 1 | 1000 | 0.00 | -10.16 | -10.16 | 0.16 | 3561930 | 0.0003 | wait_for_1h_label | 1h label is not mature yet |
| SOL | book_trade_divergence_watch | -1 | 250 | 0.00 | -10.16 | -10.16 | 0.15 | 199392 | 0.0013 | wait_for_1h_label | 1h label is not mature yet |
| HYPE | aligned_pressure_watch | 1 | 100 | 0.00 | -10.17 | -10.17 | 0.16 | 86446 | 0.0012 | wait_for_1h_label | 1h label is not mature yet |
| SOL | book_trade_divergence_watch | -1 | 500 | 0.00 | -10.17 | -10.17 | 0.15 | 199392 | 0.0025 | wait_for_1h_label | 1h label is not mature yet |
| HYPE | aligned_pressure_watch | 1 | 250 | 0.00 | -10.19 | -10.19 | 0.16 | 86446 | 0.0029 | wait_for_1h_label | 1h label is not mature yet |
| SOL | book_trade_divergence_watch | -1 | 1000 | 0.00 | -10.20 | -10.20 | 0.15 | 199392 | 0.0050 | wait_for_1h_label | 1h label is not mature yet |
| HYPE | aligned_pressure_watch | 1 | 500 | 0.00 | -10.22 | -10.22 | 0.16 | 86446 | 0.0058 | wait_for_1h_label | 1h label is not mature yet |
| HYPE | aligned_pressure_watch | 1 | 1000 | 0.00 | -10.27 | -10.27 | 0.16 | 86446 | 0.0116 | wait_for_1h_label | 1h label is not mature yet |
| XMR | book_trade_divergence_watch | 1 | 100 | 0.00 | -10.45 | -10.45 | 0.32 | 7495 | 0.0133 | wait_for_1h_label | 1h label is not mature yet |
| NEAR | book_trade_divergence_watch | 1 | 100 | 0.00 | -10.53 | -10.53 | 0.45 | 12782 | 0.0078 | wait_for_1h_label | 1h label is not mature yet |
| SUI | book_trade_divergence_watch | 1 | 100 | 0.00 | -10.54 | -10.54 | 0.53 | 63461 | 0.0016 | wait_for_1h_label | 1h label is not mature yet |
| SUI | book_trade_divergence_watch | 1 | 250 | 0.00 | -10.57 | -10.57 | 0.53 | 63461 | 0.0039 | wait_for_1h_label | 1h label is not mature yet |
| ETH | book_trade_divergence_watch | -1 | 100 | 0.00 | -10.59 | -10.59 | 0.59 | 10650019 | 0.0000 | wait_for_1h_label | 1h label is not mature yet |
| ETH | book_trade_divergence_watch | -1 | 250 | 0.00 | -10.59 | -10.59 | 0.59 | 10650019 | 0.0000 | wait_for_1h_label | 1h label is not mature yet |
| ETH | book_trade_divergence_watch | -1 | 500 | 0.00 | -10.59 | -10.59 | 0.59 | 10650019 | 0.0000 | wait_for_1h_label | 1h label is not mature yet |
| ETH | book_trade_divergence_watch | -1 | 1000 | 0.00 | -10.59 | -10.59 | 0.59 | 10650019 | 0.0001 | wait_for_1h_label | 1h label is not mature yet |

## Interpretation

`microstructure_small_paper_probe` means the 15m and 1h directional labels survived a rough taker-fee, spread, and visible-depth check. It still needs real fill logs, queue/adverse-selection measurement, and repeat snapshots before it can be treated as a trading edge.
