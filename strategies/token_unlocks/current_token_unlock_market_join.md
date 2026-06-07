# Current Token Unlock Market Join

This joins scheduled unlock events to current Hyperliquid perp context. It is not a trade instruction.

| symbol | name | unlock action | in | value USD | % supply | funding | OI notional | action | score |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| ME | Magic Eden | unlock_supply_shock_watch | 2 | 10430000.00 | 30.8000 | -0.37705405 | 170443.71424800 | unlock_supply_shock_crowded_short_overlap | 53.429777 |
| PYTH | Pyth Network | unlock_context | 346 | 93650000.00 | 37.0000 | 0.10950000 | 609511.70142000 | unlock_context | 49.933617 |
| ZRO | LayerZero | unlock_supply_shock_watch | 12 | 34450000.00 | 10.2000 | 0.25601012 | 25232698.36429400 | unlock_short_pressure_funding_overlap | 29.179543 |
| KAITO | KAITO | unlock_supply_shock_watch | 12 | 7950000.00 | 7.3000 | 0.10880971 | 852772.11230000 | unlock_short_pressure_funding_overlap | 23.186711 |
| PIXEL | Pixels | unlock_supply_shock_watch | 11 | 630970.00 | 11.8000 | 0.00000000 | 0.00000000 | unlock_supply_shock_watch | 21.583991 |
| HYPE | Hyperliquid | large_unlock_watch | 28 | 594900000.00 | 4.5000 | 0.10950000 | 1256835665.19535875 | large_unlock_watch | 21.448286 |
| AI | Sleepless AI | unlock_supply_shock_watch | 23 | 444360.00 | 12.7000 | 0.00000000 | 0.00000000 | unlock_supply_shock_watch | 20.965486 |
| MOVE | Movement | unlock_context | 2 | 1950000.00 | 4.3000 | 0.10950000 | 264694.64965200 | unlock_context | 20.500558 |
| EIGEN | EigenCloud (prev. EigenLayer) | unlock_supply_shock_watch | 23 | 7750000.00 | 5.0000 | 0.10950000 | 3859619.84269400 | unlock_short_pressure_funding_overlap | 19.156320 |
| LINEA | Linea | unlock_context | 3 | 2690000.00 | 3.6000 | 0.10950000 | 1505262.97787400 | unlock_context | 18.358996 |
| SOPH | Sophon | unlock_supply_shock_watch | 21 | 1210000.00 | 5.2000 | 0.10950000 | 87080.56491400 | unlock_short_pressure_funding_overlap | 17.610880 |
| CYBER | CYBER | unlock_supply_shock_watch | 7 | 2780000.00 | 5.9000 | 0.00000000 | 0.00000000 | unlock_supply_shock_watch | 17.609097 |
| BABY | Babylon | unlock_context | 2 | 2140000.00 | 3.7000 | -0.06606179 | 571762.89475000 | unlock_context | 16.644659 |
| IO | io.net | unlock_context | 4 | 1830000.00 | 3.8000 | -0.50639545 | 415886.54704800 | unlock_context | 16.190576 |
| NIL | Nillion | unlock_context | 17 | 1250000.00 | 4.3000 | -0.83217284 | 817484.95296000 | unlock_context | 15.835848 |
| ZORA | Zora | unlock_context | 22 | 1660000.00 | 3.7000 | -0.11055032 | 597091.29398800 | unlock_context | 15.595368 |
| XPL | Plasma | unlock_context | 18 | 7310000.00 | 3.4000 | 0.10950000 | 14679954.81435000 | unlock_context | 15.465501 |
| ALT | AltLayer | unlock_context | 17 | 1820000.00 | 3.9000 | -0.72024370 | 217446.32368800 | unlock_context | 15.362045 |
| ZETA | ZetaChain | unlock_context | 23 | 2210000.00 | 3.1000 | 0.10950000 | 375002.04976200 | unlock_context | 14.745391 |
| APT | Aptos | unlock_context | 4 | 7610000.00 | 1.4000 | 0.10950000 | 4222668.94414400 | unlock_context | 12.712547 |
| ZK | ZKsync | unlock_context | 11 | 2750000.00 | 1.8000 | 0.10950000 | 725764.73280000 | unlock_context | 12.468711 |
| OP | Optimism | unlock_context | 22 | 3780000.00 | 1.5000 | 0.07523526 | 2954254.76997200 | unlock_context | 11.481235 |
| ACE | Fusionist | unlock_context | 10 | 161390.00 | 1.8000 | 0.10950000 | 160707.45878400 | unlock_context | 10.731791 |
| W | Wormhole | unlock_context | 5 | 483600.00 | 0.8000 | 0.03643810 | 745090.62928400 | unlock_context | 8.968190 |
| IOTA | IOTA | unlock_context | 2 | 559600.00 | 0.3000 | 0.08794164 | 247018.66140000 | unlock_context | 7.285598 |

## Interpretation

Tradable unlock candidates need forward labels around the unlock window. Current perp funding and OI only show whether the event overlaps a liquid venue.
