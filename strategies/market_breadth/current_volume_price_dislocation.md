# Current Volume Price Dislocation

This scans broad CoinGecko market data for volume-backed reversal, continuation, and chase-risk candidates. It is a candidate-generation screen, not a trade list.

| symbol | name | status | side | score | rank | vol/mcap | 24h | 7d | 30d | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| HOME | HOME | capitulation_reversal_watch | watch_reversal_trigger | 55.3500 | 253 | 1.4789 | -5.46 | -43.05 | 94.77 | wait for HOME reversal trigger, then label capitulation rebound |
| CHZ | Chiliz | volume_reversal_candidate | long_reversal | 54.8344 | 138 | 0.2802 | 4.92 | -27.76 | -41.95 | paper-label CHZ volume-backed reversal over 1h, 4h, 12h, and 24h |
| WLD | Worldcoin | chase_risk | wait_or_fade_watch | 51.6025 | 47 | 0.5129 | 5.35 | 15.82 | 83.81 | avoid chasing WLD; label pullback or fade setup first |
| BCH | Bitcoin Cash | capitulation_reversal_watch | watch_reversal_trigger | 50.5595 | 26 | 0.1459 | -9.26 | -28.11 | -53.61 | wait for BCH reversal trigger, then label capitulation rebound |
| NEAR | NEAR Protocol | volume_reversal_candidate | long_reversal | 49.5604 | 37 | 0.2150 | 4.54 | -18.97 | 36.03 | paper-label NEAR volume-backed reversal over 1h, 4h, 12h, and 24h |
| VELVET | Velvet | chase_risk | wait_or_fade_watch | 47.8296 | 221 | 0.2313 | 24.20 | 238.94 | 248.86 | avoid chasing VELVET; label pullback or fade setup first |
| SEI | Sei | capitulation_reversal_watch | watch_reversal_trigger | 47.7811 | 128 | 0.1503 | -0.48 | -30.16 | -30.27 | wait for SEI reversal trigger, then label capitulation rebound |
| IP | Story | capitulation_reversal_watch | watch_reversal_trigger | 47.3452 | 259 | 0.2429 | -2.90 | -30.72 | -47.96 | wait for IP reversal trigger, then label capitulation rebound |
| ADA | Cardano | volume_reversal_candidate | long_reversal | 46.9398 | 17 | 0.0711 | 3.52 | -26.05 | -37.48 | paper-label ADA volume-backed reversal over 1h, 4h, 12h, and 24h |
| BTW | Bitway | chase_risk | wait_or_fade_watch | 44.7460 | 222 | 0.1808 | -2.89 | 398.06 | 340.00 | avoid chasing BTW; label pullback or fade setup first |
| INJ | Injective | volume_reversal_candidate | long_reversal | 44.6504 | 94 | 0.1852 | 3.66 | -19.58 | 33.55 | paper-label INJ volume-backed reversal over 1h, 4h, 12h, and 24h |
| BILL | Billions Network | capitulation_reversal_watch | watch_reversal_trigger | 44.1728 | 190 | 0.2514 | -6.64 | -23.59 | -42.79 | wait for BILL reversal trigger, then label capitulation rebound |
| OP | Optimism | capitulation_reversal_watch | watch_reversal_trigger | 43.4573 | 171 | 0.2395 | -1.19 | -22.64 | -41.60 | wait for OP reversal trigger, then label capitulation rebound |
| ZEC | Zcash | volume_reversal_candidate | long_reversal | 42.6109 | 15 | 0.1246 | 4.84 | -16.04 | -23.45 | paper-label ZEC volume-backed reversal over 1h, 4h, 12h, and 24h |
| FET | Artificial Superintelligence Alliance | capitulation_reversal_watch | watch_reversal_trigger | 42.6068 | 106 | 0.1710 | -1.34 | -22.65 | -10.45 | wait for FET reversal trigger, then label capitulation rebound |
| ZRO | LayerZero | capitulation_reversal_watch | watch_reversal_trigger | 42.0467 | 169 | 0.1576 | -6.41 | -26.04 | -42.22 | wait for ZRO reversal trigger, then label capitulation rebound |
| AAVE | Aave | capitulation_reversal_watch | watch_reversal_trigger | 41.1238 | 71 | 0.1451 | -0.35 | -20.97 | -34.04 | wait for AAVE reversal trigger, then label capitulation rebound |
| NIGHT | Midnight | volume_reversal_candidate | long_reversal | 39.0464 | 97 | 0.0589 | 9.11 | -16.25 | 0.42 | paper-label NIGHT volume-backed reversal over 1h, 4h, 12h, and 24h |
| HYPE | Hyperliquid | volume_reversal_candidate | long_reversal | 38.6459 | 10 | 0.0663 | 6.92 | -13.25 | 48.20 | paper-label HYPE volume-backed reversal over 1h, 4h, 12h, and 24h |
| RENDER | Render | capitulation_reversal_watch | watch_reversal_trigger | 38.3693 | 78 | 0.0803 | -1.94 | -22.45 | -17.97 | wait for RENDER reversal trigger, then label capitulation rebound |
| SAND | The Sandbox | capitulation_reversal_watch | watch_reversal_trigger | 36.6912 | 214 | 0.1305 | -1.82 | -24.56 | -34.71 | wait for SAND reversal trigger, then label capitulation rebound |
| BAT | Basic Attention | volume_reversal_candidate | long_reversal | 36.1690 | 213 | 0.1087 | 10.47 | -14.83 | -12.19 | paper-label BAT volume-backed reversal over 1h, 4h, 12h, and 24h |
| PUMP | Pump.fun | volume_reversal_candidate | long_reversal | 35.8282 | 93 | 0.1190 | 6.63 | -11.72 | -24.21 | paper-label PUMP volume-backed reversal over 1h, 4h, 12h, and 24h |
| NXPC | Nexpace | volume_reversal_candidate | long_reversal | 34.9751 | 266 | 0.1381 | 9.35 | -15.64 | 6.18 | paper-label NXPC volume-backed reversal over 1h, 4h, 12h, and 24h |
| ZK | ZKsync | capitulation_reversal_watch | watch_reversal_trigger | 34.4186 | 269 | 0.0992 | -0.55 | -26.92 | -45.16 | wait for ZK reversal trigger, then label capitulation rebound |
| LUNC | Terra Luna Classic | volume_reversal_candidate | long_reversal | 34.0769 | 117 | 0.1061 | 7.23 | -11.33 | -24.37 | paper-label LUNC volume-backed reversal over 1h, 4h, 12h, and 24h |
| GWEI | ETHGas | chase_risk | wait_or_fade_watch | 33.7677 | 124 | 0.1271 | 33.21 | 69.37 | 33.54 | avoid chasing GWEI; label pullback or fade setup first |
| APE | ApeCoin | volume_reversal_candidate | long_reversal | 30.5449 | 232 | 0.1555 | 3.37 | -14.45 | -22.63 | paper-label APE volume-backed reversal over 1h, 4h, 12h, and 24h |
| MANA | Decentraland | capitulation_reversal_watch | watch_reversal_trigger | 29.7885 | 224 | 0.0864 | -0.83 | -20.80 | -31.25 | wait for MANA reversal trigger, then label capitulation rebound |
| TWT | Trust Wallet | volume_reversal_candidate | long_reversal | 25.5387 | 197 | 0.0474 | 3.57 | -13.98 | -17.71 | paper-label TWT volume-backed reversal over 1h, 4h, 12h, and 24h |

## Interpretation

`volume_reversal_candidate` looks for heavy-volume rebound after a weak 7d move. `capitulation_reversal_watch` is a falling setup that still needs a trigger. `breakout_continuation_watch` is already moving and needs stop/entry discipline. `chase_risk` should usually be avoided until pullback or fade labels exist.
