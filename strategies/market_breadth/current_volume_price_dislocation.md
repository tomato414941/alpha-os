# Current Volume Price Dislocation

This scans broad CoinGecko market data for volume-backed reversal, continuation, and chase-risk candidates. It is a candidate-generation screen, not a trade list.

| symbol | name | status | side | score | rank | vol/mcap | 24h | 7d | 30d | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ZEC | Zcash | volume_reversal_candidate | long_reversal | 50.8332 | 15 | 0.1436 | 5.30 | -22.67 | -30.28 | paper-label ZEC volume-backed reversal over 1h, 4h, 12h, and 24h |
| XPL | Plasma | volume_reversal_candidate | long_reversal | 50.1608 | 182 | 0.3417 | 3.63 | -22.63 | -34.69 | paper-label XPL volume-backed reversal over 1h, 4h, 12h, and 24h |
| HOME | HOME | capitulation_reversal_watch | watch_reversal_trigger | 49.4593 | 234 | 1.5910 | -10.55 | -28.16 | 125.72 | wait for HOME reversal trigger, then label capitulation rebound |
| IP | Story | volume_reversal_candidate | long_reversal | 49.3362 | 254 | 0.3448 | 4.04 | -27.07 | -44.55 | paper-label IP volume-backed reversal over 1h, 4h, 12h, and 24h |
| ALLO | Allora | chase_risk | wait_or_fade_watch | 49.0000 | 280 | 3.3538 | 45.26 | 137.52 | 314.58 | avoid chasing ALLO; label pullback or fade setup first |
| BTW | Bitway | chase_risk | wait_or_fade_watch | 48.3487 | 226 | 0.2441 | 2.18 | 385.97 | 333.36 | avoid chasing BTW; label pullback or fade setup first |
| FET | Artificial Superintelligence Alliance | volume_reversal_candidate | long_reversal | 48.2888 | 107 | 0.2003 | 3.42 | -23.20 | -9.77 | paper-label FET volume-backed reversal over 1h, 4h, 12h, and 24h |
| BCH | Bitcoin Cash | capitulation_reversal_watch | watch_reversal_trigger | 48.2206 | 26 | 0.1303 | -6.79 | -26.70 | -54.24 | wait for BCH reversal trigger, then label capitulation rebound |
| INJ | Injective | volume_reversal_candidate | long_reversal | 47.7268 | 94 | 0.1823 | 11.51 | -14.97 | 34.41 | paper-label INJ volume-backed reversal over 1h, 4h, 12h, and 24h |
| SIREN | Siren | chase_risk | wait_or_fade_watch | 47.4192 | 80 | 0.1422 | -5.03 | 111.55 | -7.31 | avoid chasing SIREN; label pullback or fade setup first |
| FARTCOIN | Fartcoin | volume_reversal_candidate | long_reversal | 43.8178 | 245 | 0.1912 | 6.25 | -23.35 | -52.98 | paper-label FARTCOIN volume-backed reversal over 1h, 4h, 12h, and 24h |
| VELVET | Velvet | chase_risk | wait_or_fade_watch | 43.5076 | 220 | 0.1585 | 49.99 | 247.35 | 264.30 | avoid chasing VELVET; label pullback or fade setup first |
| VIRTUAL | Virtuals Protocol | volume_reversal_candidate | long_reversal | 43.5046 | 121 | 0.2034 | 4.96 | -17.39 | -37.40 | paper-label VIRTUAL volume-backed reversal over 1h, 4h, 12h, and 24h |
| OP | Optimism | volume_reversal_candidate | long_reversal | 42.9748 | 172 | 0.2687 | 5.45 | -15.00 | -39.53 | paper-label OP volume-backed reversal over 1h, 4h, 12h, and 24h |
| NIGHT | Midnight | volume_reversal_candidate | long_reversal | 42.1753 | 100 | 0.1201 | 8.29 | -16.68 | -2.51 | paper-label NIGHT volume-backed reversal over 1h, 4h, 12h, and 24h |
| SUI | Sui | volume_reversal_candidate | long_reversal | 41.6090 | 31 | 0.1813 | 3.50 | -13.78 | -28.13 | paper-label SUI volume-backed reversal over 1h, 4h, 12h, and 24h |
| SOL | Solana | volume_reversal_candidate | long_reversal | 41.3919 | 7 | 0.0817 | 3.87 | -17.97 | -28.99 | paper-label SOL volume-backed reversal over 1h, 4h, 12h, and 24h |
| PEPE | Pepe | volume_reversal_candidate | long_reversal | 41.2294 | 63 | 0.1626 | 3.13 | -16.49 | -34.78 | paper-label PEPE volume-backed reversal over 1h, 4h, 12h, and 24h |
| EIGEN | EigenCloud (prev. EigenLayer) | volume_reversal_candidate | long_reversal | 40.5111 | 215 | 0.2373 | 8.97 | -13.06 | -15.07 | paper-label EIGEN volume-backed reversal over 1h, 4h, 12h, and 24h |
| HYPE | Hyperliquid | volume_reversal_candidate | long_reversal | 40.4606 | 10 | 0.0552 | 6.85 | -15.80 | 41.31 | paper-label HYPE volume-backed reversal over 1h, 4h, 12h, and 24h |
| TRUMP | Official Trump | volume_reversal_candidate | long_reversal | 39.5517 | 116 | 0.1864 | 4.45 | -14.72 | -32.21 | paper-label TRUMP volume-backed reversal over 1h, 4h, 12h, and 24h |
| ETH | Ethereum | volume_reversal_candidate | long_reversal | 39.2375 | 2 | 0.0810 | 4.28 | -15.20 | -27.46 | paper-label ETH volume-backed reversal over 1h, 4h, 12h, and 24h |
| BAT | Basic Attention | volume_reversal_candidate | long_reversal | 38.8181 | 219 | 0.0836 | 5.38 | -24.37 | -19.57 | paper-label BAT volume-backed reversal over 1h, 4h, 12h, and 24h |
| LTC | Litecoin | volume_reversal_candidate | long_reversal | 38.3859 | 30 | 0.1024 | 3.02 | -15.73 | -26.11 | paper-label LTC volume-backed reversal over 1h, 4h, 12h, and 24h |
| WIF | dogwifhat | volume_reversal_candidate | long_reversal | 37.3413 | 196 | 0.2541 | 3.71 | -13.19 | -28.77 | paper-label WIF volume-backed reversal over 1h, 4h, 12h, and 24h |
| FIL | Filecoin | volume_reversal_candidate | long_reversal | 37.2026 | 90 | 0.1274 | 3.44 | -15.62 | -36.64 | paper-label FIL volume-backed reversal over 1h, 4h, 12h, and 24h |
| ZK | ZKsync | volume_reversal_candidate | long_reversal | 36.0904 | 265 | 0.1044 | 3.10 | -24.98 | -44.67 | paper-label ZK volume-backed reversal over 1h, 4h, 12h, and 24h |
| LUNC | Terra Luna Classic | volume_reversal_candidate | long_reversal | 36.0660 | 119 | 0.0914 | 6.16 | -15.37 | -22.76 | paper-label LUNC volume-backed reversal over 1h, 4h, 12h, and 24h |
| SKYAI | SkyAI | chase_risk | wait_or_fade_watch | 35.8713 | 142 | 0.2096 | 7.71 | 61.57 | -56.71 | avoid chasing SKYAI; label pullback or fade setup first |
| BONK | Bonk | volume_reversal_candidate | long_reversal | 35.3920 | 118 | 0.0746 | 3.13 | -18.68 | -38.51 | paper-label BONK volume-backed reversal over 1h, 4h, 12h, and 24h |
| DOGE | Dogecoin | volume_reversal_candidate | long_reversal | 35.0234 | 11 | 0.0584 | 3.30 | -13.77 | -21.27 | paper-label DOGE volume-backed reversal over 1h, 4h, 12h, and 24h |
| JUP | Jupiter | volume_reversal_candidate | long_reversal | 34.7883 | 101 | 0.0470 | 3.86 | -18.16 | -35.06 | paper-label JUP volume-backed reversal over 1h, 4h, 12h, and 24h |
| AXS | Axie Infinity | volume_reversal_candidate | long_reversal | 34.7643 | 192 | 0.1229 | 3.39 | -18.60 | -35.03 | paper-label AXS volume-backed reversal over 1h, 4h, 12h, and 24h |
| DYDX | dYdX | volume_reversal_candidate | long_reversal | 34.7082 | 240 | 0.0815 | 4.86 | -21.96 | -19.16 | paper-label DYDX volume-backed reversal over 1h, 4h, 12h, and 24h |
| LDO | Lido DAO | volume_reversal_candidate | long_reversal | 32.8616 | 159 | 0.1103 | 3.14 | -16.06 | -32.45 | paper-label LDO volume-backed reversal over 1h, 4h, 12h, and 24h |
| LINK | Chainlink | volume_reversal_candidate | long_reversal | 32.8406 | 20 | 0.0542 | 3.77 | -11.82 | -24.20 | paper-label LINK volume-backed reversal over 1h, 4h, 12h, and 24h |
| IMX | Immutable | volume_reversal_candidate | long_reversal | 32.7685 | 241 | 0.1229 | 11.16 | -11.29 | -25.80 | paper-label IMX volume-backed reversal over 1h, 4h, 12h, and 24h |
| RAY | Raydium | volume_reversal_candidate | long_reversal | 32.5975 | 199 | 0.1287 | 3.38 | -16.44 | -30.54 | paper-label RAY volume-backed reversal over 1h, 4h, 12h, and 24h |
| IOTA | IOTA | volume_reversal_candidate | long_reversal | 31.9719 | 170 | 0.0427 | 6.04 | -16.87 | -22.34 | paper-label IOTA volume-backed reversal over 1h, 4h, 12h, and 24h |
| ENS | Ethereum Name Service | volume_reversal_candidate | long_reversal | 31.7247 | 177 | 0.0852 | 4.13 | -16.33 | -30.19 | paper-label ENS volume-backed reversal over 1h, 4h, 12h, and 24h |
| GALA | GALA | volume_reversal_candidate | long_reversal | 30.6775 | 231 | 0.1510 | 3.63 | -14.54 | -37.62 | paper-label GALA volume-backed reversal over 1h, 4h, 12h, and 24h |
| GWEI | ETHGas | chase_risk | wait_or_fade_watch | 30.1440 | 131 | 0.1469 | 14.17 | 51.52 | 16.58 | avoid chasing GWEI; label pullback or fade setup first |
| CVX | Convex Finance | volume_reversal_candidate | long_reversal | 29.1769 | 251 | 0.0619 | 3.84 | -19.17 | -28.55 | paper-label CVX volume-backed reversal over 1h, 4h, 12h, and 24h |
| AKT | Akash Network | volume_reversal_candidate | long_reversal | 29.0557 | 178 | 0.0423 | 8.01 | -12.41 | -14.77 | paper-label AKT volume-backed reversal over 1h, 4h, 12h, and 24h |
| FLOKI | FLOKI | volume_reversal_candidate | long_reversal | 28.4493 | 156 | 0.0950 | 3.45 | -12.10 | -31.48 | paper-label FLOKI volume-backed reversal over 1h, 4h, 12h, and 24h |
| APE | ApeCoin | volume_reversal_candidate | long_reversal | 28.0572 | 227 | 0.1453 | 5.39 | -10.30 | -22.69 | paper-label APE volume-backed reversal over 1h, 4h, 12h, and 24h |
| 1INCH | 1INCH | volume_reversal_candidate | long_reversal | 27.3419 | 272 | 0.1037 | 3.29 | -16.43 | -28.92 | paper-label 1INCH volume-backed reversal over 1h, 4h, 12h, and 24h |

## Interpretation

`volume_reversal_candidate` looks for heavy-volume rebound after a weak 7d move. `capitulation_reversal_watch` is a falling setup that still needs a trigger. `breakout_continuation_watch` is already moving and needs stop/entry discipline. `chase_risk` should usually be avoided until pullback or fade labels exist.
