# Current L2 Imbalance Paper Gate

This subtracts taker round-trip fees and current spread from the book-imbalance directional label, then checks visible 10 bps depth. It is a directional paper gate, not a maker-fill model.

| asset | size USD | imbalance10 | cost bps | net15 bps | net1h bps | depth USD | depth usage | gate | reason |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ETH | 100 | -0.2573 | 10.59 |  |  | 6256698 | 0.0000 | wait_for_label | no 15m imbalance label yet |
| BTC | 100 | 0.2567 | 10.16 |  |  | 2773318 | 0.0000 | wait_for_label | no 15m imbalance label yet |
| ETH | 250 | -0.2573 | 10.59 |  |  | 6256698 | 0.0000 | wait_for_label | no 15m imbalance label yet |
| ETH | 500 | -0.2573 | 10.59 |  |  | 6256698 | 0.0001 | wait_for_label | no 15m imbalance label yet |
| BTC | 250 | 0.2567 | 10.16 |  |  | 2773318 | 0.0001 | wait_for_label | no 15m imbalance label yet |
| ETH | 1000 | -0.2573 | 10.59 |  |  | 6256698 | 0.0002 | wait_for_label | no 15m imbalance label yet |
| BTC | 500 | 0.2567 | 10.16 |  |  | 2773318 | 0.0002 | wait_for_label | no 15m imbalance label yet |
| SOL | 100 | 0.0284 | 10.15 |  |  | 450490 | 0.0002 | wait_for_label | no 15m imbalance label yet |
| BTC | 1000 | 0.2567 | 10.16 |  |  | 2773318 | 0.0004 | wait_for_label | no 15m imbalance label yet |
| ETH | 2500 | -0.2573 | 10.59 |  |  | 6256698 | 0.0004 | wait_for_label | no 15m imbalance label yet |
| SOL | 250 | 0.0284 | 10.15 |  |  | 450490 | 0.0006 | wait_for_label | no 15m imbalance label yet |
| ETH | 5000 | -0.2573 | 10.59 |  |  | 6256698 | 0.0008 | wait_for_label | no 15m imbalance label yet |
| BTC | 2500 | 0.2567 | 10.16 |  |  | 2773318 | 0.0009 | wait_for_label | no 15m imbalance label yet |
| HYPE | 100 | 0.3753 | 11.42 |  |  | 96552 | 0.0010 | wait_for_label | no 15m imbalance label yet |
| SOL | 500 | 0.0284 | 10.15 |  |  | 450490 | 0.0011 | wait_for_label | no 15m imbalance label yet |
| BTC | 5000 | 0.2567 | 10.16 |  |  | 2773318 | 0.0018 | wait_for_label | no 15m imbalance label yet |
| SOL | 1000 | 0.0284 | 10.15 |  |  | 450490 | 0.0022 | wait_for_label | no 15m imbalance label yet |
| HYPE | 250 | 0.3753 | 11.42 |  |  | 96552 | 0.0026 | wait_for_label | no 15m imbalance label yet |
| HYPE | 500 | 0.3753 | 11.42 |  |  | 96552 | 0.0052 | wait_for_label | no 15m imbalance label yet |
| SOL | 2500 | 0.0284 | 10.15 |  |  | 450490 | 0.0055 | wait_for_label | no 15m imbalance label yet |

## Interpretation

`small_paper_probe` means the imbalance direction survived the rough fee/spread/depth check at that notional. This does not prove a market making edge because queue position, fill probability, rebates, and adverse selection are still unmodeled.
