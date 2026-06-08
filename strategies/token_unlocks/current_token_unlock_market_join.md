# Current Token Unlock Market Join

This joins scheduled unlock events to current Hyperliquid perp context. It is not a trade instruction.

| symbol | name | unlock action | in | value USD | % supply | funding | OI notional | action | score |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| ME | Magic Eden | unlock_supply_shock_watch | 1 | 10360000.00 | 30.9000 | -0.95318962 | 205161.29536200 | unlock_supply_shock_crowded_short_overlap | 60.135162 |
| PYTH | Pyth Network | unlock_context | 345 | 93650000.00 | 37.0000 | 0.01800443 | 609768.29688400 | unlock_context | 49.670655 |
| ZRO | LayerZero | unlock_supply_shock_watch | 11 | 34450000.00 | 10.2000 | 0.10950000 | 24857138.65152000 | unlock_short_pressure_funding_overlap | 29.187340 |
| KAITO | KAITO | unlock_supply_shock_watch | 11 | 7950000.00 | 7.3000 | 0.10950000 | 866955.50148000 | unlock_short_pressure_funding_overlap | 23.306912 |
| PIXEL | Pixels | unlock_supply_shock_watch | 10 | 630970.00 | 11.8000 | 0.00000000 | 0.00000000 | unlock_supply_shock_watch | 21.734928 |
| AI | Sleepless AI | unlock_supply_shock_watch | 22 | 444360.00 | 12.7000 | 0.00000000 | 0.00000000 | unlock_supply_shock_watch | 21.001948 |
| MOVE | Movement | unlock_context | 1 | 1930000.00 | 4.3000 | -2.87776512 | 530073.95671200 | unlock_context | 20.983115 |
| HYPE | Hyperliquid | large_unlock_watch | 27 | 594900000.00 | 4.5000 | 0.10950000 | 1387579598.56272292 | large_unlock_watch | 20.962038 |
| LINEA | Linea | unlock_context | 2 | 2740000.00 | 3.6000 | 0.10950000 | 1544839.98162000 | unlock_context | 19.395864 |
| BABY | Babylon | unlock_context | 1 | 2270000.00 | 3.7000 | -0.50522950 | 472199.75943400 | unlock_context | 19.290938 |
| EIGEN | EigenCloud (prev. EigenLayer) | unlock_supply_shock_watch | 23 | 7750000.00 | 5.0000 | 0.10950000 | 3995153.64536800 | unlock_short_pressure_funding_overlap | 19.156320 |
| SOPH | Sophon | unlock_supply_shock_watch | 20 | 1210000.00 | 5.2000 | 0.10950000 | 92717.99091200 | unlock_short_pressure_funding_overlap | 17.640903 |
| IO | io.net | unlock_context | 3 | 1830000.00 | 3.8000 | 0.07317316 | 483472.08250000 | unlock_context | 16.133240 |
| ZORA | Zora | unlock_context | 21 | 1660000.00 | 3.7000 | 0.10950000 | 592045.45690000 | unlock_context | 15.617963 |
| XPL | Plasma | unlock_context | 17 | 7310000.00 | 3.4000 | 0.10950000 | 14619693.09565800 | unlock_context | 15.160192 |
| NIL | Nillion | unlock_context | 16 | 1250000.00 | 4.3000 | -0.03011863 | 897743.03262000 | unlock_context | 14.958109 |
| ZETA | ZetaChain | unlock_context | 22 | 2210000.00 | 3.1000 | 0.10950000 | 393263.61531600 | unlock_context | 14.765627 |
| ALT | AltLayer | unlock_context | 16 | 1820000.00 | 3.9000 | -0.22407642 | 226590.34498800 | unlock_context | 14.609443 |
| APT | Aptos | unlock_context | 3 | 7580000.00 | 1.4000 | -0.29968661 | 4488710.32556800 | unlock_context | 12.372643 |
| OP | Optimism | unlock_context | 21 | 3780000.00 | 1.5000 | -0.56179457 | 3182764.99713600 | unlock_context | 11.983714 |
| ZK | ZKsync | unlock_context | 10 | 2750000.00 | 1.8000 | -0.03183384 | 960364.91611800 | unlock_context | 10.849909 |
| ACE | Fusionist | unlock_context | 9 | 161850.00 | 1.8000 | 0.10950000 | 157651.33847000 | unlock_context | 10.797134 |
| W | Wormhole | unlock_context | 4 | 480330.00 | 0.8000 | -0.40746439 | 750881.41464600 | unlock_context | 9.505533 |
| CYBER | CYBER | unlock_context | 6 | 737670.00 | 1.6000 | 0.00000000 | 0.00000000 | unlock_context | 8.482656 |
| IOTA | IOTA | unlock_context | 1 | 577720.00 | 0.3000 | 0.10950000 | 242645.80473200 | unlock_context | 7.843234 |

## Interpretation

Tradable unlock candidates need forward labels around the unlock window. Current perp funding and OI only show whether the event overlaps a liquid venue.
