# Current Token Unlock Market Join

This joins scheduled unlock events to current Hyperliquid perp context. It is not a trade instruction.

| symbol | name | unlock action | in | value USD | % supply | funding | OI notional | action | score |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| ME | Magic Eden | unlock_supply_shock_watch | 2 | 10430000.00 | 30.8000 | -0.41118827 | 174131.77059600 | unlock_supply_shock_crowded_short_overlap | 53.444060 |
| PYTH | Pyth Network | unlock_context | 346 | 93650000.00 | 37.0000 | 0.10950000 | 608523.61833000 | unlock_context | 49.893296 |
| ZRO | LayerZero | unlock_supply_shock_watch | 12 | 34450000.00 | 10.2000 | 0.28941638 | 25363778.97362399 | unlock_short_pressure_funding_overlap | 29.212949 |
| KAITO | KAITO | unlock_supply_shock_watch | 12 | 7950000.00 | 7.3000 | 0.10950000 | 855983.20232000 | unlock_short_pressure_funding_overlap | 23.187401 |
| MOVE | Movement | unlock_context | 1 | 1950000.00 | 4.3000 | 0.05697592 | 260669.37342800 | unlock_context | 22.621916 |
| PIXEL | Pixels | unlock_supply_shock_watch | 11 | 630970.00 | 11.8000 | 0.00000000 | 0.00000000 | unlock_supply_shock_watch | 21.583991 |
| HYPE | Hyperliquid | large_unlock_watch | 28 | 594900000.00 | 4.5000 | 0.10950000 | 1264511699.04617977 | large_unlock_watch | 21.435101 |
| AI | Sleepless AI | unlock_supply_shock_watch | 23 | 444360.00 | 12.7000 | 0.00000000 | 0.00000000 | unlock_supply_shock_watch | 20.965486 |
| LINEA | Linea | unlock_context | 2 | 2690000.00 | 3.6000 | 0.10950000 | 1518558.93819200 | unlock_context | 19.375630 |
| EIGEN | EigenCloud (prev. EigenLayer) | unlock_supply_shock_watch | 23 | 7750000.00 | 5.0000 | 0.10950000 | 3932436.09989600 | unlock_short_pressure_funding_overlap | 19.156320 |
| BABY | Babylon | unlock_context | 2 | 2140000.00 | 3.7000 | -1.61873500 | 563700.25206600 | unlock_context | 17.666608 |
| SOPH | Sophon | unlock_supply_shock_watch | 21 | 1210000.00 | 5.2000 | 0.10950000 | 87703.14517200 | unlock_short_pressure_funding_overlap | 17.610880 |
| CYBER | CYBER | unlock_supply_shock_watch | 7 | 2780000.00 | 5.9000 | 0.00000000 | 0.00000000 | unlock_supply_shock_watch | 17.609097 |
| ZORA | Zora | unlock_context | 22 | 1660000.00 | 3.7000 | -0.94787054 | 595003.80017200 | unlock_context | 16.432688 |
| IO | io.net | unlock_context | 3 | 1830000.00 | 3.8000 | -0.16649782 | 427220.77030000 | unlock_context | 16.296701 |
| XPL | Plasma | unlock_context | 18 | 7310000.00 | 3.4000 | 0.10950000 | 14678565.31500000 | unlock_context | 15.303847 |
| NIL | Nillion | unlock_context | 17 | 1250000.00 | 4.3000 | -0.24722735 | 814734.98032800 | unlock_context | 15.129775 |
| ZETA | ZetaChain | unlock_context | 23 | 2210000.00 | 3.1000 | 0.10950000 | 379782.17848000 | unlock_context | 14.745391 |
| ALT | AltLayer | unlock_context | 17 | 1820000.00 | 3.9000 | -0.11283668 | 222689.30312200 | unlock_context | 14.566031 |
| ZK | ZKsync | unlock_context | 11 | 2750000.00 | 1.8000 | -1.02033414 | 783574.42020000 | unlock_context | 12.737701 |
| APT | Aptos | unlock_context | 4 | 7610000.00 | 1.4000 | 0.10950000 | 4174824.25878400 | unlock_context | 12.253367 |
| OP | Optimism | unlock_context | 22 | 3780000.00 | 1.5000 | 0.10950000 | 3085003.80896000 | unlock_context | 11.515499 |
| ACE | Fusionist | unlock_context | 10 | 161390.00 | 1.8000 | 0.10950000 | 160148.85069800 | unlock_context | 10.703332 |
| W | Wormhole | unlock_context | 4 | 483600.00 | 0.8000 | 0.10950000 | 743322.09100800 | unlock_context | 9.210731 |
| IOTA | IOTA | unlock_context | 2 | 559600.00 | 0.3000 | 0.10950000 | 245490.44656400 | unlock_context | 7.307156 |

## Interpretation

Tradable unlock candidates need forward labels around the unlock window. Current perp funding and OI only show whether the event overlaps a liquid venue.
