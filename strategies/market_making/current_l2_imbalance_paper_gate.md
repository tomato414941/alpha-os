# Current L2 Imbalance Paper Gate

This subtracts taker round-trip fees and current spread from the book-imbalance directional label, then checks visible 10 bps depth. It is a directional paper gate, not a maker-fill model.

| asset | size USD | imbalance10 | cost bps | net15 bps | net1h bps | depth USD | depth usage | gate | reason |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ETH | 100 | 0.1352 | 10.61 |  |  | 11976159 | 0.0000 | wait_for_label | no 15m imbalance label yet |
| ETH | 250 | 0.1352 | 10.61 |  |  | 11976159 | 0.0000 | wait_for_label | no 15m imbalance label yet |
| BTC | 100 | 0.4525 | 10.16 |  |  | 3441674 | 0.0000 | wait_for_label | no 15m imbalance label yet |
| ETH | 500 | 0.1352 | 10.61 |  |  | 11976159 | 0.0000 | wait_for_label | no 15m imbalance label yet |
| BTC | 250 | 0.4525 | 10.16 |  |  | 3441674 | 0.0001 | wait_for_label | no 15m imbalance label yet |
| ETH | 1000 | 0.1352 | 10.61 |  |  | 11976159 | 0.0001 | wait_for_label | no 15m imbalance label yet |
| BTC | 500 | 0.4525 | 10.16 |  |  | 3441674 | 0.0001 | wait_for_label | no 15m imbalance label yet |
| XRP | 100 | -0.0966 | 10.88 |  |  | 589072 | 0.0002 | wait_for_label | no 15m imbalance label yet |
| ETH | 2500 | 0.1352 | 10.61 |  |  | 11976159 | 0.0002 | wait_for_label | no 15m imbalance label yet |
| BTC | 1000 | 0.4525 | 10.16 |  |  | 3441674 | 0.0003 | wait_for_label | no 15m imbalance label yet |
| ETH | 5000 | 0.1352 | 10.61 |  |  | 11976159 | 0.0004 | wait_for_label | no 15m imbalance label yet |
| XRP | 250 | -0.0966 | 10.88 |  |  | 589072 | 0.0004 | wait_for_label | no 15m imbalance label yet |
| SOL | 100 | -0.6055 | 10.15 |  |  | 197152 | 0.0005 | wait_for_label | no 15m imbalance label yet |
| DOGE | 100 | -0.2872 | 12.95 |  |  | 175767 | 0.0006 | wait_for_label | no 15m imbalance label yet |
| BTC | 2500 | 0.4525 | 10.16 |  |  | 3441674 | 0.0007 | wait_for_label | no 15m imbalance label yet |
| SUI | 100 | -0.3582 | 11.75 |  |  | 118401 | 0.0008 | wait_for_label | no 15m imbalance label yet |
| XRP | 500 | -0.0966 | 10.88 |  |  | 589072 | 0.0008 | wait_for_label | no 15m imbalance label yet |
| BNB | 100 | 0.0280 | 11.18 |  |  | 114603 | 0.0009 | wait_for_label | no 15m imbalance label yet |
| ADA | 100 | 0.1401 | 13.08 |  |  | 86993 | 0.0011 | wait_for_label | no 15m imbalance label yet |
| SOL | 250 | -0.6055 | 10.15 |  |  | 197152 | 0.0013 | wait_for_label | no 15m imbalance label yet |

## Interpretation

`small_paper_probe` means the imbalance direction survived the rough fee/spread/depth check at that notional. This does not prove a market making edge because queue position, fill probability, rebates, and adverse selection are still unmodeled.
