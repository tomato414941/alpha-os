# Current Volume Price Dislocation

This scans broad CoinGecko market data for volume-backed reversal, continuation, and chase-risk candidates. It is a candidate-generation screen, not a trade list.

| symbol | name | status | side | score | rank | vol/mcap | 24h | 7d | 30d | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| HOME | HOME | capitulation_reversal_watch | watch_reversal_trigger | 55.6000 | 248 | 1.4827 | -4.71 | -40.60 | 96.39 | wait for HOME reversal trigger, then label capitulation rebound |
| CHZ | Chiliz | volume_reversal_candidate | long_reversal | 54.6897 | 138 | 0.2827 | 4.63 | -28.66 | -41.91 | paper-label CHZ volume-backed reversal over 1h, 4h, 12h, and 24h |
| WLD | Worldcoin | chase_risk | wait_or_fade_watch | 51.0928 | 48 | 0.3713 | 5.94 | 15.46 | 81.97 | avoid chasing WLD; label pullback or fade setup first |
| NEAR | NEAR Protocol | volume_reversal_candidate | long_reversal | 50.4885 | 37 | 0.2173 | 3.53 | -20.77 | 36.25 | paper-label NEAR volume-backed reversal over 1h, 4h, 12h, and 24h |
| BCH | Bitcoin Cash | capitulation_reversal_watch | watch_reversal_trigger | 50.1627 | 26 | 0.1458 | -8.70 | -27.71 | -53.48 | wait for BCH reversal trigger, then label capitulation rebound |
| VELVET | Velvet | chase_risk | wait_or_fade_watch | 47.8603 | 222 | 0.2327 | 27.10 | 241.37 | 248.86 | avoid chasing VELVET; label pullback or fade setup first |
| SEI | Sei | capitulation_reversal_watch | watch_reversal_trigger | 47.8586 | 128 | 0.1506 | -0.78 | -30.22 | -29.04 | wait for SEI reversal trigger, then label capitulation rebound |
| IP | Story | capitulation_reversal_watch | watch_reversal_trigger | 47.3440 | 259 | 0.2443 | -2.43 | -30.64 | -47.46 | wait for IP reversal trigger, then label capitulation rebound |
| ADA | Cardano | volume_reversal_candidate | long_reversal | 47.2739 | 17 | 0.0829 | 3.15 | -26.03 | -37.42 | paper-label ADA volume-backed reversal over 1h, 4h, 12h, and 24h |
| INJ | Injective | volume_reversal_candidate | long_reversal | 46.9011 | 94 | 0.2103 | 3.97 | -20.02 | 33.51 | paper-label INJ volume-backed reversal over 1h, 4h, 12h, and 24h |
| APT | Aptos | capitulation_reversal_watch | watch_reversal_trigger | 45.5909 | 96 | 0.1141 | -0.40 | -28.54 | -39.75 | wait for APT reversal trigger, then label capitulation rebound |
| BTW | Bitway | chase_risk | wait_or_fade_watch | 44.9860 | 220 | 0.1831 | -8.41 | 389.98 | 336.50 | avoid chasing BTW; label pullback or fade setup first |
| BILL | Billions Network | capitulation_reversal_watch | watch_reversal_trigger | 44.1001 | 190 | 0.2551 | -6.19 | -23.30 | -45.03 | wait for BILL reversal trigger, then label capitulation rebound |
| OP | Optimism | capitulation_reversal_watch | watch_reversal_trigger | 42.7613 | 171 | 0.2403 | -1.30 | -21.90 | -41.95 | wait for OP reversal trigger, then label capitulation rebound |
| ZEC | Zcash | volume_reversal_candidate | long_reversal | 42.1714 | 15 | 0.1258 | 4.28 | -16.09 | -23.50 | paper-label ZEC volume-backed reversal over 1h, 4h, 12h, and 24h |
| FET | Artificial Superintelligence Alliance | capitulation_reversal_watch | watch_reversal_trigger | 42.1161 | 106 | 0.1723 | -1.47 | -22.08 | -9.98 | wait for FET reversal trigger, then label capitulation rebound |
| ZRO | LayerZero | capitulation_reversal_watch | watch_reversal_trigger | 41.5960 | 168 | 0.1578 | -6.28 | -25.53 | -42.02 | wait for ZRO reversal trigger, then label capitulation rebound |
| AAVE | Aave | capitulation_reversal_watch | watch_reversal_trigger | 41.0361 | 71 | 0.1473 | -0.29 | -20.75 | -33.85 | wait for AAVE reversal trigger, then label capitulation rebound |
| NIGHT | Midnight | volume_reversal_candidate | long_reversal | 39.9215 | 98 | 0.0633 | 9.38 | -16.64 | -0.05 | paper-label NIGHT volume-backed reversal over 1h, 4h, 12h, and 24h |
| HYPE | Hyperliquid | volume_reversal_candidate | long_reversal | 39.0511 | 10 | 0.0678 | 6.41 | -14.07 | 47.10 | paper-label HYPE volume-backed reversal over 1h, 4h, 12h, and 24h |
| RENDER | Render | capitulation_reversal_watch | watch_reversal_trigger | 38.6231 | 78 | 0.0808 | -2.12 | -22.68 | -18.34 | wait for RENDER reversal trigger, then label capitulation rebound |
| SAND | The Sandbox | capitulation_reversal_watch | watch_reversal_trigger | 36.6284 | 215 | 0.1324 | -1.92 | -24.43 | -35.19 | wait for SAND reversal trigger, then label capitulation rebound |
| PUMP | Pump.fun | volume_reversal_candidate | long_reversal | 35.5549 | 93 | 0.1192 | 6.40 | -11.65 | -24.90 | paper-label PUMP volume-backed reversal over 1h, 4h, 12h, and 24h |
| BAT | Basic Attention | volume_reversal_candidate | long_reversal | 35.0219 | 213 | 0.1069 | 9.33 | -14.93 | -12.29 | paper-label BAT volume-backed reversal over 1h, 4h, 12h, and 24h |
| ZK | ZKsync | capitulation_reversal_watch | watch_reversal_trigger | 34.6721 | 269 | 0.1079 | -0.17 | -26.65 | -44.98 | wait for ZK reversal trigger, then label capitulation rebound |
| NXPC | Nexpace | volume_reversal_candidate | long_reversal | 34.3482 | 266 | 0.1377 | 9.65 | -14.74 | 6.03 | paper-label NXPC volume-backed reversal over 1h, 4h, 12h, and 24h |
| LUNC | Terra Luna Classic | volume_reversal_candidate | long_reversal | 34.2103 | 117 | 0.1068 | 7.83 | -10.82 | -23.94 | paper-label LUNC volume-backed reversal over 1h, 4h, 12h, and 24h |
| GWEI | ETHGas | chase_risk | wait_or_fade_watch | 34.1059 | 124 | 0.1288 | 35.91 | 70.32 | 34.44 | avoid chasing GWEI; label pullback or fade setup first |
| APE | ApeCoin | volume_reversal_candidate | long_reversal | 31.2929 | 232 | 0.1638 | 4.16 | -13.91 | -22.62 | paper-label APE volume-backed reversal over 1h, 4h, 12h, and 24h |
| MANA | Decentraland | capitulation_reversal_watch | watch_reversal_trigger | 29.7150 | 223 | 0.0855 | -0.85 | -20.74 | -31.17 | wait for MANA reversal trigger, then label capitulation rebound |
| BANANAS31 | Banana For Scale | breakout_continuation_watch | long_momentum_watch | 28.7615 | 239 | 0.0918 | 10.88 | 9.32 | -9.60 | paper-label BANANAS31 breakout continuation and stop behavior |
| TWT | Trust Wallet | volume_reversal_candidate | long_reversal | 25.1868 | 197 | 0.0482 | 3.42 | -13.72 | -18.15 | paper-label TWT volume-backed reversal over 1h, 4h, 12h, and 24h |

## Interpretation

`volume_reversal_candidate` looks for heavy-volume rebound after a weak 7d move. `capitulation_reversal_watch` is a falling setup that still needs a trigger. `breakout_continuation_watch` is already moving and needs stop/entry discipline. `chase_risk` should usually be avoided until pullback or fade labels exist.
