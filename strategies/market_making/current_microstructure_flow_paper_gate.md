# Current Microstructure Flow Paper Gate

This subtracts taker round-trip fees, current spread, and a rough visible-depth impact from microstructure flow labels. It is a small paper-probe gate, not a maker queue or fill model.

| asset | action | dir | size USD | gross15 bps | net15 bps | net1h bps | spread bps | depth USD | usage | gate | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SOL | book_trade_divergence_watch | -1 | 100 | 0.00 | -10.15 | -10.15 | 0.15 | 345545 | 0.0003 | wait_for_1h_label | 1h label is not mature yet |
| SOL | book_trade_divergence_watch | -1 | 250 | 0.00 | -10.15 | -10.15 | 0.15 | 345545 | 0.0007 | wait_for_1h_label | 1h label is not mature yet |
| SOL | book_trade_divergence_watch | -1 | 500 | 0.00 | -10.16 | -10.16 | 0.15 | 345545 | 0.0014 | wait_for_1h_label | 1h label is not mature yet |
| SOL | book_trade_divergence_watch | -1 | 1000 | 0.00 | -10.18 | -10.18 | 0.15 | 345545 | 0.0029 | wait_for_1h_label | 1h label is not mature yet |
| SUI | book_trade_divergence_watch | 1 | 100 | 0.00 | -10.27 | -10.27 | 0.26 | 106075 | 0.0009 | wait_for_1h_label | 1h label is not mature yet |
| SUI | book_trade_divergence_watch | 1 | 250 | 0.00 | -10.28 | -10.28 | 0.26 | 106075 | 0.0024 | wait_for_1h_label | 1h label is not mature yet |
| SUI | book_trade_divergence_watch | 1 | 500 | 0.00 | -10.31 | -10.31 | 0.26 | 106075 | 0.0047 | wait_for_1h_label | 1h label is not mature yet |
| BTC | book_trade_divergence_watch | 1 | 100 | 0.00 | -10.31 | -10.31 | 0.31 | 6102437 | 0.0000 | wait_for_1h_label | 1h label is not mature yet |
| BTC | book_trade_divergence_watch | 1 | 250 | 0.00 | -10.31 | -10.31 | 0.31 | 6102437 | 0.0000 | wait_for_1h_label | 1h label is not mature yet |
| BTC | book_trade_divergence_watch | 1 | 500 | 0.00 | -10.31 | -10.31 | 0.31 | 6102437 | 0.0001 | wait_for_1h_label | 1h label is not mature yet |
| BTC | book_trade_divergence_watch | 1 | 1000 | 0.00 | -10.31 | -10.31 | 0.31 | 6102437 | 0.0002 | wait_for_1h_label | 1h label is not mature yet |
| HYPE | aligned_pressure_watch | 1 | 100 | 0.00 | -10.31 | -10.31 | 0.31 | 168362 | 0.0006 | wait_for_1h_label | 1h label is not mature yet |
| HYPE | aligned_pressure_watch | 1 | 250 | 0.00 | -10.32 | -10.32 | 0.31 | 168362 | 0.0015 | wait_for_1h_label | 1h label is not mature yet |
| HYPE | aligned_pressure_watch | 1 | 500 | 0.00 | -10.34 | -10.34 | 0.31 | 168362 | 0.0030 | wait_for_1h_label | 1h label is not mature yet |
| SUI | book_trade_divergence_watch | 1 | 1000 | 0.00 | -10.35 | -10.35 | 0.26 | 106075 | 0.0094 | wait_for_1h_label | 1h label is not mature yet |
| HYPE | aligned_pressure_watch | 1 | 1000 | 0.00 | -10.37 | -10.37 | 0.31 | 168362 | 0.0059 | wait_for_1h_label | 1h label is not mature yet |
| ETH | no_clear_pressure | 1 | 100 | 0.00 | -10.59 | -10.59 | 0.59 | 12735901 | 0.0000 | wait_for_1h_label | 1h label is not mature yet |
| ETH | no_clear_pressure | 1 | 250 | 0.00 | -10.59 | -10.59 | 0.59 | 12735901 | 0.0000 | wait_for_1h_label | 1h label is not mature yet |
| ETH | no_clear_pressure | 1 | 500 | 0.00 | -10.59 | -10.59 | 0.59 | 12735901 | 0.0000 | wait_for_1h_label | 1h label is not mature yet |
| ETH | no_clear_pressure | 1 | 1000 | 0.00 | -10.59 | -10.59 | 0.59 | 12735901 | 0.0001 | wait_for_1h_label | 1h label is not mature yet |

## Interpretation

`microstructure_small_paper_probe` means the 15m and 1h directional labels survived a rough taker-fee, spread, and visible-depth check. It still needs real fill logs, queue/adverse-selection measurement, and repeat snapshots before it can be treated as a trading edge.
