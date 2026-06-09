# Current Token Unlock Market Join

This joins scheduled unlock events to current Hyperliquid perp context. It is not a trade instruction.

| symbol | name | unlock action | in | value USD | % supply | funding | OI notional | action | score |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| ME | Magic Eden | unlock_supply_shock_watch | 1 | 10360000.00 | 30.9000 | -0.54292027 | 182371.99860000 | unlock_supply_shock_crowded_short_overlap | 59.506421 |
| PYTH | Pyth Network | unlock_context | 345 | 93650000.00 | 37.0000 | -0.23820893 | 590736.37266000 | unlock_context | 49.809355 |
| ZRO | LayerZero | unlock_supply_shock_watch | 11 | 34450000.00 | 10.2000 | 0.10950000 | 23783428.62049200 | unlock_short_pressure_funding_overlap | 29.187340 |
| MOVE | Movement | unlock_context | 0 | 1930000.00 | 4.3000 | -1.48420417 | 490092.98120000 | unlock_context | 27.353296 |
| KAITO | KAITO | unlock_supply_shock_watch | 11 | 7950000.00 | 7.3000 | 0.10950000 | 820009.14240000 | unlock_short_pressure_funding_overlap | 23.306912 |
| PIXEL | Pixels | unlock_supply_shock_watch | 10 | 630970.00 | 11.8000 | 0.00000000 | 0.00000000 | unlock_supply_shock_watch | 21.734928 |
| LINEA | Linea | unlock_context | 1 | 2740000.00 | 3.6000 | 0.10950000 | 1532520.93349800 | unlock_context | 21.431659 |
| AI | Sleepless AI | unlock_supply_shock_watch | 22 | 444360.00 | 12.7000 | 0.00000000 | 0.00000000 | unlock_supply_shock_watch | 21.001948 |
| HYPE | Hyperliquid | large_unlock_watch | 27 | 594900000.00 | 4.5000 | 0.10950000 | 1292472505.44139981 | large_unlock_watch | 20.648130 |
| BABY | Babylon | unlock_context | 1 | 2270000.00 | 3.7000 | -0.78767642 | 448705.72506800 | unlock_context | 19.665094 |
| EIGEN | EigenCloud (prev. EigenLayer) | unlock_supply_shock_watch | 22 | 7750000.00 | 5.0000 | 0.01434800 | 3802599.00251200 | unlock_short_pressure_funding_overlap | 18.633665 |
| SOPH | Sophon | unlock_supply_shock_watch | 20 | 1210000.00 | 5.2000 | 0.10950000 | 93124.44385200 | unlock_short_pressure_funding_overlap | 17.640903 |
| IO | io.net | unlock_context | 2 | 1830000.00 | 3.8000 | -0.61704914 | 458509.98455200 | unlock_context | 17.479789 |
| ZORA | Zora | unlock_context | 21 | 1660000.00 | 3.7000 | 0.10950000 | 553428.20998000 | unlock_context | 15.617963 |
| XPL | Plasma | unlock_context | 17 | 7310000.00 | 3.4000 | 0.00181682 | 13592270.24588000 | unlock_context | 15.244132 |
| NIL | Nillion | unlock_context | 16 | 1250000.00 | 4.3000 | -0.27067524 | 806476.15440000 | unlock_context | 15.129259 |
| ZETA | ZetaChain | unlock_context | 22 | 2210000.00 | 3.1000 | 0.10950000 | 377039.21075200 | unlock_context | 14.765627 |
| ALT | AltLayer | unlock_context | 16 | 1820000.00 | 3.9000 | 0.10950000 | 222849.95745600 | unlock_context | 14.417584 |
| APT | Aptos | unlock_context | 3 | 7580000.00 | 1.4000 | -0.19886864 | 4352235.91496400 | unlock_context | 12.076626 |
| OP | Optimism | unlock_context | 21 | 3780000.00 | 1.5000 | -0.12468896 | 3096600.04680000 | unlock_context | 11.546609 |
| ZK | ZKsync | unlock_context | 10 | 2750000.00 | 1.8000 | 0.09812952 | 1061904.50560000 | unlock_context | 11.004404 |
| ACE | Fusionist | unlock_context | 9 | 161850.00 | 1.8000 | 0.10950000 | 152927.39375800 | unlock_context | 10.797134 |
| W | Wormhole | unlock_context | 3 | 480330.00 | 0.8000 | -0.45219470 | 730883.96755200 | unlock_context | 9.804349 |
| CYBER | CYBER | unlock_context | 6 | 737670.00 | 1.6000 | 0.00000000 | 0.00000000 | unlock_context | 8.482656 |
| IOTA | IOTA | unlock_context | 1 | 577720.00 | 0.3000 | 0.10950000 | 235468.93070400 | unlock_context | 7.843234 |

## Interpretation

Tradable unlock candidates need forward labels around the unlock window. Current perp funding and OI only show whether the event overlaps a liquid venue.
