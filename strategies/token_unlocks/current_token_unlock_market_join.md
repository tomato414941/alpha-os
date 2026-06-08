# Current Token Unlock Market Join

This joins scheduled unlock events to current Hyperliquid perp context. It is not a trade instruction.

| symbol | name | unlock action | in | value USD | % supply | funding | OI notional | action | score |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| ME | Magic Eden | unlock_supply_shock_watch | 2 | 10360000.00 | 30.9000 | -0.70236716 | 179188.53120000 | unlock_supply_shock_crowded_short_overlap | 53.654078 |
| PYTH | Pyth Network | unlock_context | 346 | 93650000.00 | 37.0000 | 0.00343567 | 582794.50260000 | unlock_context | 49.705413 |
| ZRO | LayerZero | unlock_supply_shock_watch | 12 | 34450000.00 | 10.2000 | 0.10950000 | 24808721.11069200 | unlock_short_pressure_funding_overlap | 29.033033 |
| KAITO | KAITO | unlock_supply_shock_watch | 12 | 7950000.00 | 7.3000 | 0.10950000 | 864525.33750000 | unlock_short_pressure_funding_overlap | 23.187401 |
| PIXEL | Pixels | unlock_supply_shock_watch | 11 | 630970.00 | 11.8000 | 0.00000000 | 0.00000000 | unlock_supply_shock_watch | 21.583991 |
| MOVE | Movement | unlock_context | 1 | 1930000.00 | 4.3000 | -10.92133728 | 255923.64028000 | unlock_context | 21.282068 |
| HYPE | Hyperliquid | large_unlock_watch | 28 | 594900000.00 | 4.5000 | 0.10950000 | 1290410922.27075744 | large_unlock_watch | 21.047665 |
| AI | Sleepless AI | unlock_supply_shock_watch | 23 | 444360.00 | 12.7000 | 0.00000000 | 0.00000000 | unlock_supply_shock_watch | 20.965486 |
| LINEA | Linea | unlock_context | 2 | 2740000.00 | 3.6000 | 0.10950000 | 1503646.99203200 | unlock_context | 19.395864 |
| EIGEN | EigenCloud (prev. EigenLayer) | unlock_supply_shock_watch | 23 | 7750000.00 | 5.0000 | 0.10950000 | 3991720.23445800 | unlock_short_pressure_funding_overlap | 19.156320 |
| SOPH | Sophon | unlock_supply_shock_watch | 20 | 1210000.00 | 5.2000 | 0.10950000 | 86661.07735200 | unlock_short_pressure_funding_overlap | 17.640903 |
| IO | io.net | unlock_context | 3 | 1830000.00 | 3.8000 | -1.00520825 | 416202.53180800 | unlock_context | 17.075567 |
| BABY | Babylon | unlock_context | 2 | 2270000.00 | 3.7000 | -0.18438924 | 482277.88742000 | unlock_context | 16.945175 |
| ZORA | Zora | unlock_context | 22 | 1660000.00 | 3.7000 | -0.23181413 | 569673.00976000 | unlock_context | 15.716632 |
| XPL | Plasma | unlock_context | 17 | 7310000.00 | 3.4000 | 0.06033800 | 14154079.52774400 | unlock_context | 15.346971 |
| NIL | Nillion | unlock_context | 16 | 1250000.00 | 4.3000 | -0.10277670 | 867134.64768000 | unlock_context | 15.058903 |
| ZETA | ZetaChain | unlock_context | 23 | 2210000.00 | 3.1000 | 0.10950000 | 372180.18240000 | unlock_context | 14.745391 |
| ALT | AltLayer | unlock_context | 17 | 1820000.00 | 3.9000 | 0.04427479 | 222482.95254000 | unlock_context | 14.461465 |
| APT | Aptos | unlock_context | 4 | 7580000.00 | 1.4000 | -0.38630636 | 4290075.35093200 | unlock_context | 12.325076 |
| ZK | ZKsync | unlock_context | 11 | 2750000.00 | 1.8000 | -0.56762522 | 788842.52148000 | unlock_context | 11.515488 |
| OP | Optimism | unlock_context | 22 | 3780000.00 | 1.5000 | -0.06147593 | 3099145.30070400 | unlock_context | 11.467475 |
| ACE | Fusionist | unlock_context | 10 | 161850.00 | 1.8000 | 0.10950000 | 158072.83251200 | unlock_context | 10.733600 |
| W | Wormhole | unlock_context | 4 | 480330.00 | 0.8000 | -0.59310631 | 749596.64551200 | unlock_context | 9.691175 |
| CYBER | CYBER | unlock_context | 6 | 737670.00 | 1.6000 | 0.00000000 | 0.00000000 | unlock_context | 8.482656 |
| IOTA | IOTA | unlock_context | 2 | 577720.00 | 0.3000 | 0.10950000 | 238415.37956000 | unlock_context | 7.317263 |

## Interpretation

Tradable unlock candidates need forward labels around the unlock window. Current perp funding and OI only show whether the event overlaps a liquid venue.
