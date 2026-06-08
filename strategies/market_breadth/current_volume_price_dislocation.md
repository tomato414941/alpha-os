# Current Volume Price Dislocation

This scans broad CoinGecko market data for volume-backed reversal, continuation, and chase-risk candidates. It is a candidate-generation screen, not a trade list.

| symbol | name | status | side | score | rank | vol/mcap | 24h | 7d | 30d | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ZEC | Zcash | volume_reversal_candidate | long_reversal | 63.3433 | 15 | 0.1823 | 19.87 | -23.15 | -28.54 | paper-label ZEC volume-backed reversal over 1h, 4h, 12h, and 24h |
| JTO | Jito | breakout_continuation_watch | long_momentum_watch | 60.1703 | 133 | 0.2803 | 23.26 | 18.88 | 8.03 | paper-label JTO breakout continuation and stop behavior |
| FARTCOIN | Fartcoin | volume_reversal_candidate | long_reversal | 52.4656 | 244 | 0.2626 | 8.91 | -27.74 | -54.48 | paper-label FARTCOIN volume-backed reversal over 1h, 4h, 12h, and 24h |
| XPL | Plasma | volume_reversal_candidate | long_reversal | 52.2201 | 181 | 0.3718 | 5.16 | -23.11 | -37.22 | paper-label XPL volume-backed reversal over 1h, 4h, 12h, and 24h |
| AAVE | Aave | volume_reversal_candidate | long_reversal | 50.9449 | 70 | 0.2098 | 4.34 | -22.52 | -33.47 | paper-label AAVE volume-backed reversal over 1h, 4h, 12h, and 24h |
| SIREN | Siren | chase_risk | wait_or_fade_watch | 50.7700 | 73 | 0.1570 | 50.72 | 162.19 | 16.98 | avoid chasing SIREN; label pullback or fade setup first |
| PENGU | Pudgy Penguins | volume_reversal_candidate | long_reversal | 50.3684 | 112 | 0.3127 | 10.89 | -12.07 | -33.40 | paper-label PENGU volume-backed reversal over 1h, 4h, 12h, and 24h |
| FET | Artificial Superintelligence Alliance | volume_reversal_candidate | long_reversal | 50.0619 | 107 | 0.2227 | 6.23 | -20.82 | -8.91 | paper-label FET volume-backed reversal over 1h, 4h, 12h, and 24h |
| ADA | Cardano | volume_reversal_candidate | long_reversal | 49.6354 | 18 | 0.0940 | 4.90 | -29.83 | -39.49 | paper-label ADA volume-backed reversal over 1h, 4h, 12h, and 24h |
| HOME | HOME | chase_risk | wait_or_fade_watch | 49.5828 | 238 | 0.7319 | -40.04 | -17.15 | 113.93 | avoid chasing HOME; label pullback or fade setup first |
| NEAR | NEAR Protocol | volume_reversal_candidate | long_reversal | 49.1003 | 37 | 0.2451 | 10.09 | -11.15 | 28.77 | paper-label NEAR volume-backed reversal over 1h, 4h, 12h, and 24h |
| SAHARA | Sahara AI | breakout_continuation_watch | long_momentum_watch | 48.9376 | 229 | 0.5205 | 14.79 | 12.59 | 22.31 | paper-label SAHARA breakout continuation and stop behavior |
| SEI | Sei | volume_reversal_candidate | long_reversal | 48.2817 | 129 | 0.1577 | 5.27 | -27.13 | -24.89 | paper-label SEI volume-backed reversal over 1h, 4h, 12h, and 24h |
| UB | Unibase | volume_reversal_candidate | long_reversal | 48.1271 | 131 | 0.0639 | 10.85 | -35.08 | 6.84 | paper-label UB volume-backed reversal over 1h, 4h, 12h, and 24h |
| BTW | Bitway | chase_risk | wait_or_fade_watch | 47.3086 | 214 | 0.2168 | 17.16 | 412.70 | 346.10 | avoid chasing BTW; label pullback or fade setup first |
| VIRTUAL | Virtuals Protocol | volume_reversal_candidate | long_reversal | 46.7936 | 121 | 0.2011 | 6.07 | -19.71 | -38.30 | paper-label VIRTUAL volume-backed reversal over 1h, 4h, 12h, and 24h |
| SUI | Sui | volume_reversal_candidate | long_reversal | 46.5064 | 31 | 0.2081 | 5.34 | -15.23 | -25.94 | paper-label SUI volume-backed reversal over 1h, 4h, 12h, and 24h |
| SOL | Solana | volume_reversal_candidate | long_reversal | 45.8349 | 7 | 0.0845 | 6.84 | -19.28 | -27.78 | paper-label SOL volume-backed reversal over 1h, 4h, 12h, and 24h |
| PEPE | Pepe | volume_reversal_candidate | long_reversal | 45.5958 | 62 | 0.1820 | 4.89 | -17.89 | -34.71 | paper-label PEPE volume-backed reversal over 1h, 4h, 12h, and 24h |
| TAO | Bittensor | volume_reversal_candidate | long_reversal | 45.4786 | 42 | 0.0987 | 10.48 | -16.18 | -32.17 | paper-label TAO volume-backed reversal over 1h, 4h, 12h, and 24h |
| CHZ | Chiliz | capitulation_reversal_watch | watch_reversal_trigger | 45.3976 | 145 | 0.1983 | -0.10 | -25.75 | -44.36 | wait for CHZ reversal trigger, then label capitulation rebound |
| LUNC | Terra Luna Classic | volume_reversal_candidate | long_reversal | 44.9005 | 122 | 0.1013 | 6.66 | -23.26 | -27.95 | paper-label LUNC volume-backed reversal over 1h, 4h, 12h, and 24h |
| INJ | Injective | volume_reversal_candidate | long_reversal | 44.8070 | 96 | 0.2020 | 5.05 | -17.44 | 28.25 | paper-label INJ volume-backed reversal over 1h, 4h, 12h, and 24h |
| OP | Optimism | volume_reversal_candidate | long_reversal | 44.7491 | 171 | 0.2829 | 3.62 | -17.70 | -43.66 | paper-label OP volume-backed reversal over 1h, 4h, 12h, and 24h |
| TRUMP | Official Trump | volume_reversal_candidate | long_reversal | 43.4184 | 117 | 0.1933 | 5.53 | -17.14 | -32.55 | paper-label TRUMP volume-backed reversal over 1h, 4h, 12h, and 24h |
| ETH | Ethereum | volume_reversal_candidate | long_reversal | 42.6992 | 2 | 0.0756 | 7.34 | -15.92 | -26.95 | paper-label ETH volume-backed reversal over 1h, 4h, 12h, and 24h |
| ARB | Arbitrum | volume_reversal_candidate | long_reversal | 42.3990 | 101 | 0.1480 | 3.71 | -19.86 | -42.50 | paper-label ARB volume-backed reversal over 1h, 4h, 12h, and 24h |
| TIA | Celestia | volume_reversal_candidate | long_reversal | 41.7967 | 135 | 0.1322 | 5.03 | -20.59 | -28.43 | paper-label TIA volume-backed reversal over 1h, 4h, 12h, and 24h |
| LTC | Litecoin | volume_reversal_candidate | long_reversal | 41.5198 | 30 | 0.1135 | 3.32 | -17.89 | -26.49 | paper-label LTC volume-backed reversal over 1h, 4h, 12h, and 24h |
| FIL | Filecoin | volume_reversal_candidate | long_reversal | 41.4579 | 91 | 0.1341 | 5.31 | -17.65 | -40.15 | paper-label FIL volume-backed reversal over 1h, 4h, 12h, and 24h |
| RENDER | Render | volume_reversal_candidate | long_reversal | 41.1807 | 77 | 0.0856 | 4.93 | -19.97 | -17.78 | paper-label RENDER volume-backed reversal over 1h, 4h, 12h, and 24h |
| HYPE | Hyperliquid | volume_reversal_candidate | long_reversal | 40.4779 | 11 | 0.0523 | 5.12 | -17.77 | 37.74 | paper-label HYPE volume-backed reversal over 1h, 4h, 12h, and 24h |
| WIF | dogwifhat | volume_reversal_candidate | long_reversal | 40.1391 | 199 | 0.2346 | 4.00 | -17.01 | -29.97 | paper-label WIF volume-backed reversal over 1h, 4h, 12h, and 24h |
| VELVET | Velvet | chase_risk | wait_or_fade_watch | 39.9447 | 264 | 0.1357 | 34.71 | 153.35 | 173.62 | avoid chasing VELVET; label pullback or fade setup first |
| EIGEN | EigenCloud (prev. EigenLayer) | volume_reversal_candidate | long_reversal | 39.4730 | 218 | 0.1766 | 11.06 | -13.72 | -18.15 | paper-label EIGEN volume-backed reversal over 1h, 4h, 12h, and 24h |
| DYDX | dYdX | volume_reversal_candidate | long_reversal | 38.9851 | 235 | 0.0658 | 7.86 | -23.92 | -18.75 | paper-label DYDX volume-backed reversal over 1h, 4h, 12h, and 24h |
| BAT | Basic Attention | volume_reversal_candidate | long_reversal | 38.6537 | 222 | 0.0926 | 4.20 | -25.80 | -21.51 | paper-label BAT volume-backed reversal over 1h, 4h, 12h, and 24h |
| PUMP | Pump.fun | volume_reversal_candidate | long_reversal | 38.2424 | 97 | 0.1004 | 6.92 | -15.15 | -28.79 | paper-label PUMP volume-backed reversal over 1h, 4h, 12h, and 24h |
| CFG | Centrifuge | volume_reversal_candidate | long_reversal | 38.1825 | 240 | 0.0688 | 7.25 | -23.81 | -30.94 | paper-label CFG volume-backed reversal over 1h, 4h, 12h, and 24h |
| BONK | Bonk | volume_reversal_candidate | long_reversal | 37.9470 | 118 | 0.0815 | 4.54 | -19.41 | -38.32 | paper-label BONK volume-backed reversal over 1h, 4h, 12h, and 24h |
| DOT | Polkadot | volume_reversal_candidate | long_reversal | 37.6236 | 48 | 0.0679 | 3.05 | -17.90 | -29.10 | paper-label DOT volume-backed reversal over 1h, 4h, 12h, and 24h |
| LINK | Chainlink | volume_reversal_candidate | long_reversal | 37.5384 | 19 | 0.0538 | 7.01 | -13.25 | -23.38 | paper-label LINK volume-backed reversal over 1h, 4h, 12h, and 24h |
| DOGE | Dogecoin | volume_reversal_candidate | long_reversal | 37.5167 | 10 | 0.0637 | 5.21 | -13.99 | -21.28 | paper-label DOGE volume-backed reversal over 1h, 4h, 12h, and 24h |
| LDO | Lido DAO | volume_reversal_candidate | long_reversal | 37.4835 | 158 | 0.1237 | 5.79 | -17.18 | -33.63 | paper-label LDO volume-backed reversal over 1h, 4h, 12h, and 24h |
| ZRO | LayerZero | capitulation_reversal_watch | watch_reversal_trigger | 37.0452 | 159 | 0.1413 | -0.90 | -21.52 | -38.30 | wait for ZRO reversal trigger, then label capitulation rebound |
| UNI | Uniswap | volume_reversal_candidate | long_reversal | 37.0444 | 51 | 0.0918 | 4.02 | -15.07 | -30.61 | paper-label UNI volume-backed reversal over 1h, 4h, 12h, and 24h |
| SAND | The Sandbox | volume_reversal_candidate | long_reversal | 37.0124 | 213 | 0.0849 | 3.25 | -24.32 | -34.85 | paper-label SAND volume-backed reversal over 1h, 4h, 12h, and 24h |
| ETHFI | Ether.fi | volume_reversal_candidate | long_reversal | 36.9271 | 143 | 0.0943 | 4.04 | -19.38 | -33.48 | paper-label ETHFI volume-backed reversal over 1h, 4h, 12h, and 24h |
| RAY | Raydium | volume_reversal_candidate | long_reversal | 35.9395 | 201 | 0.1324 | 4.88 | -18.17 | -31.62 | paper-label RAY volume-backed reversal over 1h, 4h, 12h, and 24h |
| TWT | Trust Wallet | volume_reversal_candidate | long_reversal | 35.2550 | 205 | 0.0606 | 8.37 | -18.50 | -21.01 | paper-label TWT volume-backed reversal over 1h, 4h, 12h, and 24h |

## Interpretation

`volume_reversal_candidate` looks for heavy-volume rebound after a weak 7d move. `capitulation_reversal_watch` is a falling setup that still needs a trigger. `breakout_continuation_watch` is already moving and needs stop/entry discipline. `chase_risk` should usually be avoided until pullback or fade labels exist.
