# Current Volume Price Dislocation

This scans broad CoinGecko market data for volume-backed reversal, continuation, and chase-risk candidates. It is a candidate-generation screen, not a trade list.

| symbol | name | status | side | score | rank | vol/mcap | 24h | 7d | 30d | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ZEC | Zcash | volume_reversal_candidate | long_reversal | 55.0111 | 15 | 0.1709 | 5.99 | -24.52 | -31.42 | paper-label ZEC volume-backed reversal over 1h, 4h, 12h, and 24h |
| HOME | HOME | chase_risk | wait_or_fade_watch | 52.1500 | 217 | 0.9371 | -36.50 | -13.14 | 142.39 | avoid chasing HOME; label pullback or fade setup first |
| SIREN | Siren | chase_risk | wait_or_fade_watch | 50.7943 | 78 | 0.1616 | 40.16 | 126.83 | -9.58 | avoid chasing SIREN; label pullback or fade setup first |
| BTW | Bitway | chase_risk | wait_or_fade_watch | 50.6352 | 244 | 0.2973 | 4.74 | 312.23 | 267.06 | avoid chasing BTW; label pullback or fade setup first |
| IP | Story | capitulation_reversal_watch | watch_reversal_trigger | 48.5329 | 258 | 0.2499 | -1.49 | -31.44 | -46.99 | wait for IP reversal trigger, then label capitulation rebound |
| FET | Artificial Superintelligence Alliance | capitulation_reversal_watch | watch_reversal_trigger | 47.7664 | 108 | 0.2033 | -0.27 | -25.97 | -13.27 | wait for FET reversal trigger, then label capitulation rebound |
| CHZ | Chiliz | capitulation_reversal_watch | watch_reversal_trigger | 46.9903 | 146 | 0.2114 | -3.85 | -26.61 | -46.34 | wait for CHZ reversal trigger, then label capitulation rebound |
| APT | Aptos | capitulation_reversal_watch | watch_reversal_trigger | 46.5316 | 95 | 0.1168 | -1.71 | -29.27 | -40.48 | wait for APT reversal trigger, then label capitulation rebound |
| XPL | Plasma | capitulation_reversal_watch | watch_reversal_trigger | 46.2752 | 181 | 0.3335 | -0.12 | -22.33 | -37.19 | wait for XPL reversal trigger, then label capitulation rebound |
| SEI | Sei | capitulation_reversal_watch | watch_reversal_trigger | 44.8152 | 129 | 0.1525 | -0.89 | -27.11 | -27.90 | wait for SEI reversal trigger, then label capitulation rebound |
| AVAX | Avalanche | capitulation_reversal_watch | watch_reversal_trigger | 44.1606 | 32 | 0.0843 | -2.15 | -25.70 | -33.40 | wait for AVAX reversal trigger, then label capitulation rebound |
| XLM | Stellar | capitulation_reversal_watch | watch_reversal_trigger | 42.5923 | 16 | 0.0846 | -4.47 | -23.32 | 20.26 | wait for XLM reversal trigger, then label capitulation rebound |
| VELVET | Velvet | chase_risk | wait_or_fade_watch | 41.0240 | 248 | 0.1404 | 38.64 | 189.87 | 199.15 | avoid chasing VELVET; label pullback or fade setup first |
| SAHARA | Sahara AI | breakout_continuation_watch | long_momentum_watch | 40.4488 | 234 | 0.5274 | 10.73 | 8.42 | 17.97 | paper-label SAHARA breakout continuation and stop behavior |
| EIGEN | EigenCloud (prev. EigenLayer) | volume_reversal_candidate | long_reversal | 40.3087 | 215 | 0.2395 | 9.01 | -12.68 | -16.60 | paper-label EIGEN volume-backed reversal over 1h, 4h, 12h, and 24h |
| GRASS | Grass | capitulation_reversal_watch | watch_reversal_trigger | 40.1072 | 171 | 0.1232 | -3.76 | -26.26 | -10.66 | wait for GRASS reversal trigger, then label capitulation rebound |
| ETH | Ethereum | volume_reversal_candidate | long_reversal | 39.7707 | 2 | 0.0862 | 3.10 | -16.59 | -27.92 | paper-label ETH volume-backed reversal over 1h, 4h, 12h, and 24h |
| NIGHT | Midnight | capitulation_reversal_watch | watch_reversal_trigger | 39.2122 | 104 | 0.1329 | -4.66 | -21.44 | -9.02 | wait for NIGHT reversal trigger, then label capitulation rebound |
| HYPE | Hyperliquid | volume_reversal_candidate | long_reversal | 37.4815 | 10 | 0.0559 | 4.97 | -14.66 | 42.09 | paper-label HYPE volume-backed reversal over 1h, 4h, 12h, and 24h |
| RENDER | Render | capitulation_reversal_watch | watch_reversal_trigger | 37.4682 | 79 | 0.0828 | -1.84 | -21.45 | -18.67 | wait for RENDER reversal trigger, then label capitulation rebound |
| ZRO | LayerZero | capitulation_reversal_watch | watch_reversal_trigger | 37.0405 | 160 | 0.1336 | -3.30 | -22.02 | -40.55 | wait for ZRO reversal trigger, then label capitulation rebound |
| LUNC | Terra Luna Classic | volume_reversal_candidate | long_reversal | 36.7820 | 117 | 0.0936 | 6.85 | -15.16 | -24.12 | paper-label LUNC volume-backed reversal over 1h, 4h, 12h, and 24h |
| SAND | The Sandbox | capitulation_reversal_watch | watch_reversal_trigger | 36.5404 | 213 | 0.1036 | -1.44 | -25.97 | -37.25 | wait for SAND reversal trigger, then label capitulation rebound |
| DYDX | dYdX | volume_reversal_candidate | long_reversal | 34.9443 | 237 | 0.0720 | 4.11 | -23.37 | -19.18 | paper-label DYDX volume-backed reversal over 1h, 4h, 12h, and 24h |
| ETHFI | Ether.fi | capitulation_reversal_watch | watch_reversal_trigger | 33.6098 | 143 | 0.0840 | -1.89 | -20.72 | -34.78 | wait for ETHFI reversal trigger, then label capitulation rebound |
| PUMP | Pump.fun | volume_reversal_candidate | long_reversal | 32.4742 | 97 | 0.1019 | 3.17 | -13.04 | -29.37 | paper-label PUMP volume-backed reversal over 1h, 4h, 12h, and 24h |
| MANA | Decentraland | capitulation_reversal_watch | watch_reversal_trigger | 31.6908 | 226 | 0.0909 | -2.65 | -22.54 | -32.70 | wait for MANA reversal trigger, then label capitulation rebound |
| IMX | Immutable | volume_reversal_candidate | long_reversal | 30.9633 | 242 | 0.0947 | 6.02 | -16.37 | -28.80 | paper-label IMX volume-backed reversal over 1h, 4h, 12h, and 24h |
| SKYAI | SkyAI | chase_risk | wait_or_fade_watch | 29.8733 | 157 | 0.2045 | -12.34 | 41.81 | -64.37 | avoid chasing SKYAI; label pullback or fade setup first |
| TWT | Trust Wallet | volume_reversal_candidate | long_reversal | 28.6560 | 202 | 0.0560 | 3.01 | -17.38 | -21.21 | paper-label TWT volume-backed reversal over 1h, 4h, 12h, and 24h |
| STRK | Starknet | volume_reversal_candidate | long_reversal | 27.8395 | 168 | 0.0810 | 3.45 | -12.93 | -39.68 | paper-label STRK volume-backed reversal over 1h, 4h, 12h, and 24h |
| GWEI | ETHGas | chase_risk | wait_or_fade_watch | 25.2415 | 132 | 0.1165 | 2.20 | 39.41 | 7.20 | avoid chasing GWEI; label pullback or fade setup first |

## Interpretation

`volume_reversal_candidate` looks for heavy-volume rebound after a weak 7d move. `capitulation_reversal_watch` is a falling setup that still needs a trigger. `breakout_continuation_watch` is already moving and needs stop/entry discipline. `chase_risk` should usually be avoided until pullback or fade labels exist.
