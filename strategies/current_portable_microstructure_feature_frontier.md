# Current Portable Microstructure Feature Frontier

This checks whether the same microstructure feature shape can be compared across BTC, ETH, SOL, and HYPE. It is a feature frontier, not a trading strategy or execution instruction.

| asset | status | priority | spread | depth | book imbalance | trade imbalance | 15m | 1h | next step |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| HYPE | delayed_or_reversal_support | 104.6584 | 0.30807622 | 168362.23298000 | 0.00016578 | 1.00000000 | -0.02432653 | 0.00475881 | split HYPE horizon handling before putting this feature into a shared table |
| ETH | short_horizon_only | 102.9609 | 0.58704394 | 12735900.64988500 | -0.08243894 | 0.12099504 | 0.00723778 | -0.00172046 | keep ETH as a 15m-only feature candidate and reject 1h holding unless repeated |
| BTC | short_horizon_only | 102.3285 | 0.31161870 | 6102436.66227000 | -0.44076210 | 0.99270537 | 0.00431839 | -0.00264165 | keep BTC as a 15m-only feature candidate and reject 1h holding unless repeated |
| SOL | delayed_or_reversal_support | 96.9089 | 0.14762000 | 345545.32701000 | -0.41142717 | 0.18967921 | -0.00550215 | 0.00332547 | split SOL horizon handling before putting this feature into a shared table |
