# Current L2 Imbalance Paper Gate

This subtracts taker round-trip fees and current spread from the book-imbalance directional label, then checks visible 10 bps depth. It is a directional paper gate, not a maker-fill model.

| asset | size USD | imbalance10 | cost bps | net15 bps | net1h bps | depth USD | depth usage | gate | reason |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ETH | 100 | 0.0231 | 10.60 |  |  | 13829891 | 0.0000 | wait_for_label | no 15m imbalance label yet |
| ETH | 250 | 0.0231 | 10.60 |  |  | 13829891 | 0.0000 | wait_for_label | no 15m imbalance label yet |
| ETH | 500 | 0.0231 | 10.60 |  |  | 13829891 | 0.0000 | wait_for_label | no 15m imbalance label yet |
| BTC | 100 | 0.6875 | 10.16 |  |  | 1903045 | 0.0001 | wait_for_label | no 15m imbalance label yet |
| ETH | 1000 | 0.0231 | 10.60 |  |  | 13829891 | 0.0001 | wait_for_label | no 15m imbalance label yet |
| BTC | 250 | 0.6875 | 10.16 |  |  | 1903045 | 0.0001 | wait_for_label | no 15m imbalance label yet |
| ETH | 2500 | 0.0231 | 10.60 |  |  | 13829891 | 0.0002 | wait_for_label | no 15m imbalance label yet |
| SOL | 100 | -0.3960 | 10.15 |  |  | 401375 | 0.0002 | wait_for_label | no 15m imbalance label yet |
| BTC | 500 | 0.6875 | 10.16 |  |  | 1903045 | 0.0003 | wait_for_label | no 15m imbalance label yet |
| ETH | 5000 | 0.0231 | 10.60 |  |  | 13829891 | 0.0004 | wait_for_label | no 15m imbalance label yet |
| BTC | 1000 | 0.6875 | 10.16 |  |  | 1903045 | 0.0005 | wait_for_label | no 15m imbalance label yet |
| SOL | 250 | -0.3960 | 10.15 |  |  | 401375 | 0.0006 | wait_for_label | no 15m imbalance label yet |
| HYPE | 100 | 0.1703 | 10.16 |  |  | 108897 | 0.0009 | wait_for_label | no 15m imbalance label yet |
| SOL | 500 | -0.3960 | 10.15 |  |  | 401375 | 0.0012 | wait_for_label | no 15m imbalance label yet |
| BTC | 2500 | 0.6875 | 10.16 |  |  | 1903045 | 0.0013 | wait_for_label | no 15m imbalance label yet |
| HYPE | 250 | 0.1703 | 10.16 |  |  | 108897 | 0.0023 | wait_for_label | no 15m imbalance label yet |
| SOL | 1000 | -0.3960 | 10.15 |  |  | 401375 | 0.0025 | wait_for_label | no 15m imbalance label yet |
| BTC | 5000 | 0.6875 | 10.16 |  |  | 1903045 | 0.0026 | wait_for_label | no 15m imbalance label yet |
| HYPE | 500 | 0.1703 | 10.16 |  |  | 108897 | 0.0046 | wait_for_label | no 15m imbalance label yet |
| SOL | 2500 | -0.3960 | 10.15 |  |  | 401375 | 0.0062 | wait_for_label | no 15m imbalance label yet |

## Interpretation

`small_paper_probe` means the imbalance direction survived the rough fee/spread/depth check at that notional. This does not prove a market making edge because queue position, fill probability, rebates, and adverse selection are still unmodeled.
