# Current Microstructure Flow Paper Gate

This subtracts taker round-trip fees, current spread, and a rough visible-depth impact from microstructure flow labels. It is a small paper-probe gate, not a maker queue or fill model.

| asset | action | dir | size USD | gross15 bps | net15 bps | net1h bps | spread bps | depth USD | usage | gate | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SOL | aligned_pressure_watch | -1 | 100 | 0.00 | -10.15 | -10.15 | 0.15 | 472311 | 0.0002 | wait_for_1h_label | 1h label is not mature yet |
| SOL | aligned_pressure_watch | -1 | 250 | 0.00 | -10.15 | -10.15 | 0.15 | 472311 | 0.0005 | wait_for_1h_label | 1h label is not mature yet |
| BTC | aligned_pressure_watch | 1 | 100 | 0.00 | -10.16 | -10.16 | 0.16 | 3549315 | 0.0000 | wait_for_1h_label | 1h label is not mature yet |
| BTC | aligned_pressure_watch | 1 | 250 | 0.00 | -10.16 | -10.16 | 0.16 | 3549315 | 0.0001 | wait_for_1h_label | 1h label is not mature yet |
| BTC | aligned_pressure_watch | 1 | 500 | 0.00 | -10.16 | -10.16 | 0.16 | 3549315 | 0.0001 | wait_for_1h_label | 1h label is not mature yet |
| SOL | aligned_pressure_watch | -1 | 500 | 0.00 | -10.16 | -10.16 | 0.15 | 472311 | 0.0011 | wait_for_1h_label | 1h label is not mature yet |
| BTC | aligned_pressure_watch | 1 | 1000 | 0.00 | -10.16 | -10.16 | 0.16 | 3549315 | 0.0003 | wait_for_1h_label | 1h label is not mature yet |
| SOL | aligned_pressure_watch | -1 | 1000 | 0.00 | -10.17 | -10.17 | 0.15 | 472311 | 0.0021 | wait_for_1h_label | 1h label is not mature yet |
| HYPE | book_trade_divergence_watch | -1 | 100 | 0.00 | -10.32 | -10.32 | 0.31 | 111545 | 0.0009 | wait_for_1h_label | 1h label is not mature yet |
| HYPE | book_trade_divergence_watch | -1 | 250 | 0.00 | -10.33 | -10.33 | 0.31 | 111545 | 0.0022 | wait_for_1h_label | 1h label is not mature yet |
| HYPE | book_trade_divergence_watch | -1 | 500 | 0.00 | -10.35 | -10.35 | 0.31 | 111545 | 0.0045 | wait_for_1h_label | 1h label is not mature yet |
| HYPE | book_trade_divergence_watch | -1 | 1000 | 0.00 | -10.40 | -10.40 | 0.31 | 111545 | 0.0090 | wait_for_1h_label | 1h label is not mature yet |
| NEAR | aligned_pressure_watch | -1 | 100 | 0.00 | -10.53 | -10.53 | 0.46 | 14083 | 0.0071 | wait_for_1h_label | 1h label is not mature yet |
| ETH | book_trade_divergence_watch | -1 | 100 | 0.00 | -10.59 | -10.59 | 0.59 | 11516676 | 0.0000 | wait_for_1h_label | 1h label is not mature yet |
| ETH | book_trade_divergence_watch | -1 | 250 | 0.00 | -10.59 | -10.59 | 0.59 | 11516676 | 0.0000 | wait_for_1h_label | 1h label is not mature yet |
| ETH | book_trade_divergence_watch | -1 | 500 | 0.00 | -10.59 | -10.59 | 0.59 | 11516676 | 0.0000 | wait_for_1h_label | 1h label is not mature yet |
| ETH | book_trade_divergence_watch | -1 | 1000 | 0.00 | -10.59 | -10.59 | 0.59 | 11516676 | 0.0001 | wait_for_1h_label | 1h label is not mature yet |
| NEAR | aligned_pressure_watch | -1 | 250 | 0.00 | -10.64 | -10.64 | 0.46 | 14083 | 0.0178 | wait_for_1h_label | 1h label is not mature yet |
| BNB | book_trade_divergence_watch | -1 | 100 | 0.00 | -10.67 | -10.67 | 0.66 | 107947 | 0.0009 | wait_for_1h_label | 1h label is not mature yet |
| BNB | book_trade_divergence_watch | -1 | 250 | 0.00 | -10.69 | -10.69 | 0.66 | 107947 | 0.0023 | wait_for_1h_label | 1h label is not mature yet |

## Interpretation

`microstructure_small_paper_probe` means the 15m and 1h directional labels survived a rough taker-fee, spread, and visible-depth check. It still needs real fill logs, queue/adverse-selection measurement, and repeat snapshots before it can be treated as a trading edge.
