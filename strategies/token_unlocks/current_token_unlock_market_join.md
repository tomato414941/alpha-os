# Current Token Unlock Market Join

This joins scheduled unlock events to current Hyperliquid perp context. It is not a trade instruction.

| symbol | name | unlock action | in | value USD | % supply | funding | OI notional | action | score |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| ME | Magic Eden | unlock_supply_shock_watch | 1 | 10360000.00 | 30.9000 | -0.61421528 | 180151.12438800 | unlock_supply_shock_crowded_short_overlap | 60.071146 |
| PYTH | Pyth Network | unlock_context | 345 | 93650000.00 | 37.0000 | -0.25553884 | 583265.00626800 | unlock_context | 49.943270 |
| ZRO | LayerZero | unlock_supply_shock_watch | 11 | 34450000.00 | 10.2000 | 0.10950000 | 24797127.57004800 | unlock_short_pressure_funding_overlap | 29.187340 |
| KAITO | KAITO | unlock_supply_shock_watch | 11 | 7950000.00 | 7.3000 | 0.10950000 | 864336.53400000 | unlock_short_pressure_funding_overlap | 23.306912 |
| PIXEL | Pixels | unlock_supply_shock_watch | 10 | 630970.00 | 11.8000 | 0.00000000 | 0.00000000 | unlock_supply_shock_watch | 21.734928 |
| MOVE | Movement | unlock_context | 1 | 1930000.00 | 4.3000 | -7.95393896 | 258509.43246600 | unlock_context | 21.274745 |
| HYPE | Hyperliquid | large_unlock_watch | 27 | 594900000.00 | 4.5000 | 0.10950000 | 1290405179.89567900 | large_unlock_watch | 21.085144 |
| AI | Sleepless AI | unlock_supply_shock_watch | 22 | 444360.00 | 12.7000 | 0.00000000 | 0.00000000 | unlock_supply_shock_watch | 21.001948 |
| LINEA | Linea | unlock_context | 2 | 2740000.00 | 3.6000 | 0.10950000 | 1505401.62351000 | unlock_context | 19.395864 |
| EIGEN | EigenCloud (prev. EigenLayer) | unlock_supply_shock_watch | 23 | 7750000.00 | 5.0000 | 0.10950000 | 3988276.92586000 | unlock_short_pressure_funding_overlap | 19.156320 |
| BABY | Babylon | unlock_context | 1 | 2270000.00 | 3.7000 | 0.10950000 | 479844.68327400 | unlock_context | 18.911662 |
| SOPH | Sophon | unlock_supply_shock_watch | 20 | 1210000.00 | 5.2000 | 0.10950000 | 86858.41195200 | unlock_short_pressure_funding_overlap | 17.640903 |
| IO | io.net | unlock_context | 3 | 1830000.00 | 3.8000 | -1.06186092 | 420135.50238200 | unlock_context | 17.076406 |
| ZORA | Zora | unlock_context | 21 | 1660000.00 | 3.7000 | -0.25740647 | 570688.10376600 | unlock_context | 15.765869 |
| XPL | Plasma | unlock_context | 17 | 7310000.00 | 3.4000 | 0.10950000 | 14507972.59392800 | unlock_context | 15.395860 |
| NIL | Nillion | unlock_context | 16 | 1250000.00 | 4.3000 | 0.01698476 | 861902.62285000 | unlock_context | 14.956075 |
| ZETA | ZetaChain | unlock_context | 22 | 2210000.00 | 3.1000 | 0.10950000 | 372969.12734400 | unlock_context | 14.765627 |
| ALT | AltLayer | unlock_context | 16 | 1820000.00 | 3.9000 | -0.10952278 | 222408.75346200 | unlock_context | 14.564741 |
| APT | Aptos | unlock_context | 3 | 7580000.00 | 1.4000 | -0.11547870 | 4307166.73673400 | unlock_context | 12.437119 |
| OP | Optimism | unlock_context | 21 | 3780000.00 | 1.5000 | -0.38005698 | 3115240.17842000 | unlock_context | 11.801977 |
| ZK | ZKsync | unlock_context | 10 | 2750000.00 | 1.8000 | -0.47566274 | 789080.39205000 | unlock_context | 11.475432 |
| ACE | Fusionist | unlock_context | 9 | 161850.00 | 1.8000 | 0.10950000 | 158127.53945200 | unlock_context | 10.797134 |
| W | Wormhole | unlock_context | 4 | 480330.00 | 0.8000 | -0.55136842 | 753554.06750600 | unlock_context | 9.649437 |
| CYBER | CYBER | unlock_context | 6 | 737670.00 | 1.6000 | 0.00000000 | 0.00000000 | unlock_context | 8.482656 |
| IOTA | IOTA | unlock_context | 1 | 577720.00 | 0.3000 | 0.10950000 | 239468.48845200 | unlock_context | 7.843234 |

## Interpretation

Tradable unlock candidates need forward labels around the unlock window. Current perp funding and OI only show whether the event overlaps a liquid venue.
