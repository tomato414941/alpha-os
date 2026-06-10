# Current Microstructure Flow Paper Gate

This subtracts taker round-trip fees, current spread, and a rough visible-depth impact from microstructure flow labels. It is a small paper-probe gate, not a maker queue or fill model.

| asset | action | dir | size USD | gross15 bps | net15 bps | net1h bps | spread bps | depth USD | usage | gate | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| BTC | aligned_pressure_watch | 1 | 100 | 0.00 | -10.16 | -10.16 | 0.16 | 783015 | 0.0001 | wait_for_1h_label | 1h label is not mature yet |
| BTC | aligned_pressure_watch | 1 | 250 | 0.00 | -10.16 | -10.16 | 0.16 | 783015 | 0.0003 | wait_for_1h_label | 1h label is not mature yet |
| BTC | aligned_pressure_watch | 1 | 500 | 0.00 | -10.17 | -10.17 | 0.16 | 783015 | 0.0006 | wait_for_1h_label | 1h label is not mature yet |
| BTC | aligned_pressure_watch | 1 | 1000 | 0.00 | -10.17 | -10.17 | 0.16 | 783015 | 0.0013 | wait_for_1h_label | 1h label is not mature yet |
| HYPE | book_trade_divergence_watch | -1 | 100 | 0.00 | -10.34 | -10.34 | 0.32 | 64559 | 0.0015 | wait_for_1h_label | 1h label is not mature yet |
| HYPE | book_trade_divergence_watch | -1 | 250 | 0.00 | -10.36 | -10.36 | 0.32 | 64559 | 0.0039 | wait_for_1h_label | 1h label is not mature yet |
| HYPE | book_trade_divergence_watch | -1 | 500 | 0.00 | -10.40 | -10.40 | 0.32 | 64559 | 0.0077 | wait_for_1h_label | 1h label is not mature yet |
| HYPE | book_trade_divergence_watch | -1 | 1000 | 0.00 | -10.48 | -10.48 | 0.32 | 64559 | 0.0155 | wait_for_1h_label | 1h label is not mature yet |
| ETH | aligned_pressure_watch | 1 | 100 | 0.00 | -10.60 | -10.60 | 0.60 | 6342670 | 0.0000 | wait_for_1h_label | 1h label is not mature yet |
| ETH | aligned_pressure_watch | 1 | 250 | 0.00 | -10.60 | -10.60 | 0.60 | 6342670 | 0.0000 | wait_for_1h_label | 1h label is not mature yet |
| ETH | aligned_pressure_watch | 1 | 500 | 0.00 | -10.60 | -10.60 | 0.60 | 6342670 | 0.0001 | wait_for_1h_label | 1h label is not mature yet |
| ETH | aligned_pressure_watch | 1 | 1000 | 0.00 | -10.60 | -10.60 | 0.60 | 6342670 | 0.0002 | wait_for_1h_label | 1h label is not mature yet |
| SOL | book_trade_divergence_watch | -1 | 100 | 0.00 | -10.77 | -10.77 | 0.77 | 982337 | 0.0001 | wait_for_1h_label | 1h label is not mature yet |
| SOL | book_trade_divergence_watch | -1 | 250 | 0.00 | -10.77 | -10.77 | 0.77 | 982337 | 0.0003 | wait_for_1h_label | 1h label is not mature yet |
| SOL | book_trade_divergence_watch | -1 | 500 | 0.00 | -10.77 | -10.77 | 0.77 | 982337 | 0.0005 | wait_for_1h_label | 1h label is not mature yet |
| SOL | book_trade_divergence_watch | -1 | 1000 | 0.00 | -10.78 | -10.78 | 0.77 | 982337 | 0.0010 | wait_for_1h_label | 1h label is not mature yet |
| SUI | book_trade_divergence_watch | 1 | 100 | 0.00 | -11.38 | -11.38 | 1.36 | 63881 | 0.0016 | wait_for_1h_label | 1h label is not mature yet |
| SUI | book_trade_divergence_watch | 1 | 250 | 0.00 | -11.40 | -11.40 | 1.36 | 63881 | 0.0039 | wait_for_1h_label | 1h label is not mature yet |
| ARB | aligned_pressure_watch | -1 | 100 | 0.00 | -11.40 | -11.40 | 1.25 | 6371 | 0.0157 | wait_for_1h_label | 1h label is not mature yet |
| SUI | book_trade_divergence_watch | 1 | 500 | 0.00 | -11.44 | -11.44 | 1.36 | 63881 | 0.0078 | wait_for_1h_label | 1h label is not mature yet |

## Interpretation

`microstructure_small_paper_probe` means the 15m and 1h directional labels survived a rough taker-fee, spread, and visible-depth check. It still needs real fill logs, queue/adverse-selection measurement, and repeat snapshots before it can be treated as a trading edge.
