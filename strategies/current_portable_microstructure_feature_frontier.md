# Current Portable Microstructure Feature Frontier

This checks whether the same microstructure feature shape can be compared across BTC, ETH, SOL, and HYPE. It is a feature frontier, not a trading strategy or execution instruction.

| asset | status | priority | spread | depth | book imbalance | trade imbalance | 15m | 1h | next step |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| HYPE | delayed_or_reversal_support | 104.1400 | 0.15767026 | 86445.59626500 | 0.33904100 | 0.76367226 | -0.02432653 | 0.00475881 | split HYPE horizon handling before putting this feature into a shared table |
| ETH | short_horizon_only | 102.9478 | 0.59359511 | 10650019.02228000 | 0.14309892 | -0.99002845 | 0.00723778 | -0.00172046 | keep ETH as a 15m-only feature candidate and reject 1h holding unless repeated |
| BTC | short_horizon_only | 102.6361 | 0.15779217 | 3561929.55027000 | 0.34796982 | 0.50865460 | 0.00431839 | -0.00264165 | keep BTC as a 15m-only feature candidate and reject 1h holding unless repeated |
| SOL | delayed_or_reversal_support | 95.4458 | 0.14841309 | 199392.11158000 | 0.69201885 | -0.90549925 | -0.00550215 | 0.00332547 | split SOL horizon handling before putting this feature into a shared table |
