# Current Volume Price Dislocation

This scans broad CoinGecko market data for volume-backed reversal, continuation, and chase-risk candidates. It is a candidate-generation screen, not a trade list.

| symbol | name | status | side | score | rank | vol/mcap | 24h | 7d | 30d | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| HOME | HOME | capitulation_reversal_watch | watch_reversal_trigger | 55.3500 | 253 | 1.4821 | -5.81 | -43.79 | 92.25 | wait for HOME reversal trigger, then label capitulation rebound |
| CHZ | Chiliz | volume_reversal_candidate | long_reversal | 53.3693 | 138 | 0.2820 | 3.35 | -28.85 | -42.82 | paper-label CHZ volume-backed reversal over 1h, 4h, 12h, and 24h |
| BCH | Bitcoin Cash | capitulation_reversal_watch | watch_reversal_trigger | 51.5674 | 26 | 0.1510 | -10.09 | -28.81 | -54.06 | wait for BCH reversal trigger, then label capitulation rebound |
| WLD | Worldcoin | chase_risk | wait_or_fade_watch | 51.4480 | 48 | 0.3741 | 3.08 | 15.56 | 83.39 | avoid chasing WLD; label pullback or fade setup first |
| IP | Story | capitulation_reversal_watch | watch_reversal_trigger | 50.4036 | 258 | 0.2937 | -4.82 | -30.68 | -47.92 | wait for IP reversal trigger, then label capitulation rebound |
| XPL | Plasma | capitulation_reversal_watch | watch_reversal_trigger | 49.3486 | 182 | 0.2953 | -1.76 | -25.73 | -35.31 | wait for XPL reversal trigger, then label capitulation rebound |
| SEI | Sei | capitulation_reversal_watch | watch_reversal_trigger | 48.7590 | 129 | 0.1521 | -2.37 | -31.08 | -31.19 | wait for SEI reversal trigger, then label capitulation rebound |
| VELVET | Velvet | chase_risk | wait_or_fade_watch | 47.5771 | 217 | 0.2238 | 30.02 | 256.40 | 266.82 | avoid chasing VELVET; label pullback or fade setup first |
| APT | Aptos | capitulation_reversal_watch | watch_reversal_trigger | 46.3042 | 96 | 0.1144 | -1.12 | -29.24 | -40.32 | wait for APT reversal trigger, then label capitulation rebound |
| VIRTUAL | Virtuals Protocol | capitulation_reversal_watch | watch_reversal_trigger | 45.7286 | 121 | 0.2398 | -1.72 | -22.39 | -38.54 | wait for VIRTUAL reversal trigger, then label capitulation rebound |
| BILL | Billions Network | capitulation_reversal_watch | watch_reversal_trigger | 44.7300 | 189 | 0.2499 | -8.03 | -24.19 | -43.24 | wait for BILL reversal trigger, then label capitulation rebound |
| BTW | Bitway | chase_risk | wait_or_fade_watch | 44.6161 | 222 | 0.1786 | -1.36 | 393.82 | 336.26 | avoid chasing BTW; label pullback or fade setup first |
| OP | Optimism | capitulation_reversal_watch | watch_reversal_trigger | 44.3312 | 171 | 0.2415 | -2.40 | -23.39 | -42.17 | wait for OP reversal trigger, then label capitulation rebound |
| ZRO | LayerZero | capitulation_reversal_watch | watch_reversal_trigger | 43.2603 | 168 | 0.1606 | -8.56 | -27.02 | -42.99 | wait for ZRO reversal trigger, then label capitulation rebound |
| FET | Artificial Superintelligence Alliance | capitulation_reversal_watch | watch_reversal_trigger | 42.9255 | 107 | 0.1705 | -2.38 | -23.05 | -10.91 | wait for FET reversal trigger, then label capitulation rebound |
| ZEC | Zcash | volume_reversal_candidate | long_reversal | 42.2695 | 15 | 0.1264 | 3.26 | -17.18 | -24.48 | paper-label ZEC volume-backed reversal over 1h, 4h, 12h, and 24h |
| AAVE | Aave | capitulation_reversal_watch | watch_reversal_trigger | 41.9415 | 71 | 0.1464 | -1.74 | -21.71 | -34.65 | wait for AAVE reversal trigger, then label capitulation rebound |
| BAT | Basic Attention | volume_reversal_candidate | long_reversal | 38.5966 | 213 | 0.1506 | 10.75 | -14.46 | -11.81 | paper-label BAT volume-backed reversal over 1h, 4h, 12h, and 24h |
| NIGHT | Midnight | volume_reversal_candidate | long_reversal | 38.1486 | 98 | 0.0544 | 7.04 | -17.74 | -1.37 | paper-label NIGHT volume-backed reversal over 1h, 4h, 12h, and 24h |
| SAND | The Sandbox | capitulation_reversal_watch | watch_reversal_trigger | 37.6131 | 216 | 0.1335 | -3.27 | -25.40 | -35.44 | wait for SAND reversal trigger, then label capitulation rebound |
| FARTCOIN | Fartcoin | capitulation_reversal_watch | watch_reversal_trigger | 37.2483 | 241 | 0.1583 | -0.21 | -24.80 | -54.99 | wait for FARTCOIN reversal trigger, then label capitulation rebound |
| ARB | Arbitrum | capitulation_reversal_watch | watch_reversal_trigger | 35.5921 | 102 | 0.0874 | -2.42 | -20.45 | -42.96 | wait for ARB reversal trigger, then label capitulation rebound |
| TIA | Celestia | capitulation_reversal_watch | watch_reversal_trigger | 35.4620 | 134 | 0.0989 | -1.45 | -21.23 | -29.33 | wait for TIA reversal trigger, then label capitulation rebound |
| PUMP | Pump.fun | volume_reversal_candidate | long_reversal | 34.9089 | 93 | 0.1213 | 4.24 | -13.04 | -25.35 | paper-label PUMP volume-backed reversal over 1h, 4h, 12h, and 24h |
| LDO | Lido DAO | capitulation_reversal_watch | watch_reversal_trigger | 33.6125 | 159 | 0.1002 | -3.26 | -20.55 | -34.10 | wait for LDO reversal trigger, then label capitulation rebound |
| LUNC | Terra Luna Classic | volume_reversal_candidate | long_reversal | 33.2095 | 117 | 0.1087 | 5.21 | -12.32 | -25.22 | paper-label LUNC volume-backed reversal over 1h, 4h, 12h, and 24h |
| GWEI | ETHGas | chase_risk | wait_or_fade_watch | 32.8865 | 124 | 0.1076 | 31.63 | 70.52 | 34.45 | avoid chasing GWEI; label pullback or fade setup first |
| AXS | Axie Infinity | capitulation_reversal_watch | watch_reversal_trigger | 32.2860 | 192 | 0.1123 | -0.93 | -20.15 | -35.15 | wait for AXS reversal trigger, then label capitulation rebound |
| NXPC | Nexpace | volume_reversal_candidate | long_reversal | 30.8541 | 266 | 0.1028 | 7.61 | -15.38 | 6.51 | paper-label NXPC volume-backed reversal over 1h, 4h, 12h, and 24h |
| MANA | Decentraland | capitulation_reversal_watch | watch_reversal_trigger | 30.7100 | 227 | 0.0877 | -2.53 | -21.80 | -32.11 | wait for MANA reversal trigger, then label capitulation rebound |

## Interpretation

`volume_reversal_candidate` looks for heavy-volume rebound after a weak 7d move. `capitulation_reversal_watch` is a falling setup that still needs a trigger. `breakout_continuation_watch` is already moving and needs stop/entry discipline. `chase_risk` should usually be avoided until pullback or fade labels exist.
