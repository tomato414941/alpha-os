# Current Token Unlock Market Join

This joins scheduled unlock events to current Hyperliquid perp context. It is not a trade instruction.

| symbol | name | unlock action | in | value USD | % supply | funding | OI notional | action | score |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| ME | Magic Eden | unlock_supply_shock_watch | 1 | 10360000.00 | 30.9000 | -0.14802823 | 164002.93057800 | unlock_supply_shock_crowded_short_overlap | 59.115256 |
| PYTH | Pyth Network | unlock_context | 345 | 93650000.00 | 37.0000 | -0.33543004 | 626901.76824000 | unlock_context | 49.969975 |
| ZRO | LayerZero | unlock_supply_shock_watch | 11 | 34450000.00 | 10.2000 | 0.10950000 | 25279516.76582400 | unlock_short_pressure_funding_overlap | 29.187340 |
| KAITO | KAITO | unlock_supply_shock_watch | 11 | 7950000.00 | 7.3000 | -0.07434612 | 863862.30000000 | unlock_supply_shock_crowded_short_overlap | 23.271759 |
| PIXEL | Pixels | unlock_supply_shock_watch | 10 | 630970.00 | 11.8000 | 0.00000000 | 0.00000000 | unlock_supply_shock_watch | 21.734928 |
| AI | Sleepless AI | unlock_supply_shock_watch | 22 | 444360.00 | 12.7000 | 0.00000000 | 0.00000000 | unlock_supply_shock_watch | 21.001948 |
| MOVE | Movement | unlock_context | 1 | 1930000.00 | 4.3000 | -1.01632732 | 489027.33809200 | unlock_context | 20.877329 |
| HYPE | Hyperliquid | large_unlock_watch | 27 | 594900000.00 | 4.5000 | 0.10950000 | 1338665127.77087927 | large_unlock_watch | 20.671574 |
| LINEA | Linea | unlock_context | 2 | 2740000.00 | 3.6000 | 0.10950000 | 1556956.79022600 | unlock_context | 19.395864 |
| EIGEN | EigenCloud (prev. EigenLayer) | unlock_supply_shock_watch | 22 | 7750000.00 | 5.0000 | 0.10950000 | 3922759.93274400 | unlock_short_pressure_funding_overlap | 19.184227 |
| BABY | Babylon | unlock_context | 1 | 2270000.00 | 3.7000 | -0.18913190 | 475388.10892000 | unlock_context | 19.032788 |
| SOPH | Sophon | unlock_supply_shock_watch | 20 | 1210000.00 | 5.2000 | 0.10950000 | 92965.53062400 | unlock_short_pressure_funding_overlap | 17.640903 |
| IO | io.net | unlock_context | 3 | 1830000.00 | 3.8000 | -1.41607064 | 488615.10285000 | unlock_context | 16.978036 |
| ZORA | Zora | unlock_context | 21 | 1660000.00 | 3.7000 | 0.10950000 | 602150.56930000 | unlock_context | 15.617963 |
| XPL | Plasma | unlock_context | 17 | 7310000.00 | 3.4000 | 0.10950000 | 14876863.05600000 | unlock_context | 15.334385 |
| NIL | Nillion | unlock_context | 16 | 1250000.00 | 4.3000 | -0.14047536 | 884425.42675200 | unlock_context | 15.036839 |
| ZETA | ZetaChain | unlock_context | 22 | 2210000.00 | 3.1000 | 0.10950000 | 393908.94426600 | unlock_context | 14.765627 |
| ALT | AltLayer | unlock_context | 16 | 1820000.00 | 3.9000 | -0.42319122 | 221014.66945200 | unlock_context | 14.710985 |
| APT | Aptos | unlock_context | 3 | 7580000.00 | 1.4000 | -0.03017119 | 4515672.25824000 | unlock_context | 11.922265 |
| OP | Optimism | unlock_context | 21 | 3780000.00 | 1.5000 | 0.10950000 | 3124784.63203200 | unlock_context | 11.531420 |
| ZK | ZKsync | unlock_context | 10 | 2750000.00 | 1.8000 | 0.10950000 | 1092941.90100000 | unlock_context | 10.942739 |
| ACE | Fusionist | unlock_context | 9 | 161850.00 | 1.8000 | 0.10950000 | 157268.37149200 | unlock_context | 10.797134 |
| W | Wormhole | unlock_context | 4 | 480330.00 | 0.8000 | -0.16989144 | 747919.96283400 | unlock_context | 9.267960 |
| CYBER | CYBER | unlock_context | 6 | 737670.00 | 1.6000 | 0.00000000 | 0.00000000 | unlock_context | 8.482656 |
| IOTA | IOTA | unlock_context | 1 | 577720.00 | 0.3000 | 0.10950000 | 242262.43208000 | unlock_context | 7.843234 |

## Interpretation

Tradable unlock candidates need forward labels around the unlock window. Current perp funding and OI only show whether the event overlaps a liquid venue.
