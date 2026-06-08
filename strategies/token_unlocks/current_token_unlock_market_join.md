# Current Token Unlock Market Join

This joins scheduled unlock events to current Hyperliquid perp context. It is not a trade instruction.

| symbol | name | unlock action | in | value USD | % supply | funding | OI notional | action | score |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| ME | Magic Eden | unlock_supply_shock_watch | 1 | 10360000.00 | 30.9000 | -0.38971225 | 202836.98675200 | unlock_supply_shock_crowded_short_overlap | 59.566075 |
| PYTH | Pyth Network | unlock_context | 345 | 93650000.00 | 37.0000 | -0.17714560 | 613309.93120000 | unlock_context | 49.818413 |
| ZRO | LayerZero | unlock_supply_shock_watch | 11 | 34450000.00 | 10.2000 | 0.10950000 | 24715278.34384400 | unlock_short_pressure_funding_overlap | 29.187340 |
| KAITO | KAITO | unlock_supply_shock_watch | 11 | 7950000.00 | 7.3000 | 0.01591955 | 868316.46456000 | unlock_short_pressure_funding_overlap | 23.213332 |
| PIXEL | Pixels | unlock_supply_shock_watch | 10 | 630970.00 | 11.8000 | 0.00000000 | 0.00000000 | unlock_supply_shock_watch | 21.734928 |
| HYPE | Hyperliquid | large_unlock_watch | 27 | 594900000.00 | 4.5000 | 0.28021050 | 1388825401.95999956 | large_unlock_watch | 21.097117 |
| AI | Sleepless AI | unlock_supply_shock_watch | 22 | 444360.00 | 12.7000 | 0.00000000 | 0.00000000 | unlock_supply_shock_watch | 21.001948 |
| MOVE | Movement | unlock_context | 1 | 1930000.00 | 4.3000 | -1.73576772 | 529290.79432800 | unlock_context | 20.957389 |
| LINEA | Linea | unlock_context | 2 | 2740000.00 | 3.6000 | 0.10950000 | 1553481.85947200 | unlock_context | 19.395864 |
| EIGEN | EigenCloud (prev. EigenLayer) | unlock_supply_shock_watch | 23 | 7750000.00 | 5.0000 | 0.10950000 | 3991027.23891200 | unlock_short_pressure_funding_overlap | 19.156320 |
| BABY | Babylon | unlock_context | 1 | 2270000.00 | 3.7000 | -0.32492942 | 478632.80350000 | unlock_context | 19.118070 |
| SOPH | Sophon | unlock_supply_shock_watch | 20 | 1210000.00 | 5.2000 | 0.10950000 | 93310.23872800 | unlock_short_pressure_funding_overlap | 17.640903 |
| IO | io.net | unlock_context | 3 | 1830000.00 | 3.8000 | -0.44863114 | 483093.72454000 | unlock_context | 16.604763 |
| ZORA | Zora | unlock_context | 21 | 1660000.00 | 3.7000 | 0.10950000 | 596644.33310600 | unlock_context | 15.617963 |
| NIL | Nillion | unlock_context | 16 | 1250000.00 | 4.3000 | -0.37968293 | 902258.54140800 | unlock_context | 15.301128 |
| XPL | Plasma | unlock_context | 17 | 7310000.00 | 3.4000 | 0.10950000 | 14920801.80451200 | unlock_context | 15.253063 |
| ZETA | ZetaChain | unlock_context | 22 | 2210000.00 | 3.1000 | 0.10950000 | 394134.76095000 | unlock_context | 14.765627 |
| ALT | AltLayer | unlock_context | 16 | 1820000.00 | 3.9000 | -0.07108214 | 221689.30291200 | unlock_context | 14.384215 |
| APT | Aptos | unlock_context | 3 | 7580000.00 | 1.4000 | -0.36992779 | 4549698.52160000 | unlock_context | 12.468665 |
| OP | Optimism | unlock_context | 21 | 3780000.00 | 1.5000 | -0.04521912 | 3197505.04762000 | unlock_context | 11.467139 |
| ZK | ZKsync | unlock_context | 10 | 2750000.00 | 1.8000 | 0.10950000 | 1061505.38174800 | unlock_context | 10.953295 |
| ACE | Fusionist | unlock_context | 9 | 161850.00 | 1.8000 | 0.10950000 | 157134.25551600 | unlock_context | 10.797134 |
| W | Wormhole | unlock_context | 4 | 480330.00 | 0.8000 | 0.01175680 | 760000.12078000 | unlock_context | 9.109825 |
| CYBER | CYBER | unlock_context | 6 | 737670.00 | 1.6000 | 0.00000000 | 0.00000000 | unlock_context | 8.482656 |
| IOTA | IOTA | unlock_context | 1 | 577720.00 | 0.3000 | 0.10950000 | 244825.22160000 | unlock_context | 7.843234 |

## Interpretation

Tradable unlock candidates need forward labels around the unlock window. Current perp funding and OI only show whether the event overlaps a liquid venue.
