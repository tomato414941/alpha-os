# Current L2 Imbalance Paper Gate

This subtracts taker round-trip fees and current spread from the book-imbalance directional label, then checks visible 10 bps depth. It is a directional paper gate, not a maker-fill model.

| asset | size USD | imbalance10 | cost bps | net15 bps | net1h bps | depth USD | depth usage | gate | reason |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ETH | 100 | 0.1378 | 10.59 |  |  | 9998689 | 0.0000 | wait_for_label | no 15m imbalance label yet |
| ETH | 250 | 0.1378 | 10.59 |  |  | 9998689 | 0.0000 | wait_for_label | no 15m imbalance label yet |
| ETH | 500 | 0.1378 | 10.59 |  |  | 9998689 | 0.0001 | wait_for_label | no 15m imbalance label yet |
| BTC | 100 | 0.7403 | 10.16 |  |  | 1341285 | 0.0001 | wait_for_label | no 15m imbalance label yet |
| ETH | 1000 | 0.1378 | 10.59 |  |  | 9998689 | 0.0001 | wait_for_label | no 15m imbalance label yet |
| XRP | 100 | -0.0873 | 11.74 |  |  | 594144 | 0.0002 | wait_for_label | no 15m imbalance label yet |
| BTC | 250 | 0.7403 | 10.16 |  |  | 1341285 | 0.0002 | wait_for_label | no 15m imbalance label yet |
| ETH | 2500 | 0.1378 | 10.59 |  |  | 9998689 | 0.0003 | wait_for_label | no 15m imbalance label yet |
| SOL | 100 | -0.5921 | 10.15 |  |  | 323153 | 0.0003 | wait_for_label | no 15m imbalance label yet |
| BTC | 500 | 0.7403 | 10.16 |  |  | 1341285 | 0.0004 | wait_for_label | no 15m imbalance label yet |
| XRP | 250 | -0.0873 | 11.74 |  |  | 594144 | 0.0004 | wait_for_label | no 15m imbalance label yet |
| ETH | 5000 | 0.1378 | 10.59 |  |  | 9998689 | 0.0005 | wait_for_label | no 15m imbalance label yet |
| DOGE | 100 | 0.0843 | 11.05 |  |  | 173677 | 0.0006 | wait_for_label | no 15m imbalance label yet |
| BTC | 1000 | 0.7403 | 10.16 |  |  | 1341285 | 0.0007 | wait_for_label | no 15m imbalance label yet |
| SOL | 250 | -0.5921 | 10.15 |  |  | 323153 | 0.0008 | wait_for_label | no 15m imbalance label yet |
| XRP | 500 | -0.0873 | 11.74 |  |  | 594144 | 0.0008 | wait_for_label | no 15m imbalance label yet |
| HYPE | 100 | -0.3833 | 10.17 |  |  | 116167 | 0.0009 | wait_for_label | no 15m imbalance label yet |
| ADA | 100 | -0.0752 | 11.82 |  |  | 76829 | 0.0013 | wait_for_label | no 15m imbalance label yet |
| DOGE | 250 | 0.0843 | 11.05 |  |  | 173677 | 0.0014 | wait_for_label | no 15m imbalance label yet |
| ZEC | 100 | -0.1615 | 11.15 |  |  | 66677 | 0.0015 | wait_for_label | no 15m imbalance label yet |

## Interpretation

`small_paper_probe` means the imbalance direction survived the rough fee/spread/depth check at that notional. This does not prove a market making edge because queue position, fill probability, rebates, and adverse selection are still unmodeled.
