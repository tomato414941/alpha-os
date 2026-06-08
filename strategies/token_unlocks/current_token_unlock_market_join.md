# Current Token Unlock Market Join

This joins scheduled unlock events to current Hyperliquid perp context. It is not a trade instruction.

| symbol | name | unlock action | in | value USD | % supply | funding | OI notional | action | score |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| ME | Magic Eden | unlock_supply_shock_watch | 2 | 10430000.00 | 30.8000 | 0.02297573 | 174576.14542200 | unlock_short_pressure_funding_overlap | 53.061751 |
| PYTH | Pyth Network | unlock_context | 346 | 93650000.00 | 37.0000 | -0.02150317 | 610308.61992800 | unlock_context | 49.815529 |
| ZRO | LayerZero | unlock_supply_shock_watch | 12 | 34450000.00 | 10.2000 | 0.11050828 | 25476502.82920800 | unlock_short_pressure_funding_overlap | 29.034041 |
| KAITO | KAITO | unlock_supply_shock_watch | 12 | 7950000.00 | 7.3000 | 0.10950000 | 858081.74256000 | unlock_short_pressure_funding_overlap | 23.187401 |
| MOVE | Movement | unlock_context | 1 | 1950000.00 | 4.3000 | -0.23322186 | 260117.63870400 | unlock_context | 22.798162 |
| PIXEL | Pixels | unlock_supply_shock_watch | 11 | 630970.00 | 11.8000 | 0.00000000 | 0.00000000 | unlock_supply_shock_watch | 21.583991 |
| HYPE | Hyperliquid | large_unlock_watch | 28 | 594900000.00 | 4.5000 | 0.10950000 | 1282179531.54768014 | large_unlock_watch | 21.371833 |
| AI | Sleepless AI | unlock_supply_shock_watch | 23 | 444360.00 | 12.7000 | 0.00000000 | 0.00000000 | unlock_supply_shock_watch | 20.965486 |
| LINEA | Linea | unlock_context | 2 | 2690000.00 | 3.6000 | 0.10950000 | 1511432.05958400 | unlock_context | 19.375630 |
| EIGEN | EigenCloud (prev. EigenLayer) | unlock_supply_shock_watch | 23 | 7750000.00 | 5.0000 | 0.10950000 | 3926495.97371000 | unlock_short_pressure_funding_overlap | 19.156320 |
| BABY | Babylon | unlock_context | 2 | 2140000.00 | 3.7000 | -0.73286861 | 551325.66494000 | unlock_context | 17.611515 |
| SOPH | Sophon | unlock_supply_shock_watch | 21 | 1210000.00 | 5.2000 | 0.10950000 | 85236.11776000 | unlock_short_pressure_funding_overlap | 17.610880 |
| CYBER | CYBER | unlock_supply_shock_watch | 7 | 2780000.00 | 5.9000 | 0.00000000 | 0.00000000 | unlock_supply_shock_watch | 17.609097 |
| IO | io.net | unlock_context | 3 | 1830000.00 | 3.8000 | -0.12661967 | 425425.33813000 | unlock_context | 16.255717 |
| ZORA | Zora | unlock_context | 22 | 1660000.00 | 3.7000 | -0.40688098 | 593947.91053000 | unlock_context | 15.891699 |
| XPL | Plasma | unlock_context | 18 | 7310000.00 | 3.4000 | 0.10950000 | 14726867.04981600 | unlock_context | 15.317405 |
| NIL | Nillion | unlock_context | 17 | 1250000.00 | 4.3000 | -0.23278912 | 810174.82958800 | unlock_context | 15.091042 |
| ZETA | ZetaChain | unlock_context | 23 | 2210000.00 | 3.1000 | 0.10950000 | 379079.25566400 | unlock_context | 14.745391 |
| ALT | AltLayer | unlock_context | 17 | 1820000.00 | 3.9000 | -0.14872115 | 225508.99240000 | unlock_context | 14.660482 |
| ZK | ZKsync | unlock_context | 11 | 2750000.00 | 1.8000 | -0.64933325 | 774419.21775000 | unlock_context | 12.401697 |
| APT | Aptos | unlock_context | 4 | 7610000.00 | 1.4000 | 0.10950000 | 4156184.08889200 | unlock_context | 12.260433 |
| OP | Optimism | unlock_context | 22 | 3780000.00 | 1.5000 | 0.10950000 | 3053886.45870000 | unlock_context | 11.515499 |
| ACE | Fusionist | unlock_context | 10 | 161390.00 | 1.8000 | 0.10950000 | 159113.11831200 | unlock_context | 10.731791 |
| W | Wormhole | unlock_context | 4 | 483600.00 | 0.8000 | 0.08740991 | 736313.61764800 | unlock_context | 9.188641 |
| IOTA | IOTA | unlock_context | 2 | 559600.00 | 0.3000 | 0.10950000 | 238956.31880000 | unlock_context | 7.307156 |

## Interpretation

Tradable unlock candidates need forward labels around the unlock window. Current perp funding and OI only show whether the event overlaps a liquid venue.
