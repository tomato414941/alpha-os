# Current Volume Price Dislocation

This scans broad CoinGecko market data for volume-backed reversal, continuation, and chase-risk candidates. It is a candidate-generation screen, not a trade list.

| symbol | name | status | side | score | rank | vol/mcap | 24h | 7d | 30d | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| HOME | HOME | capitulation_reversal_watch | watch_reversal_trigger | 49.2522 | 232 | 1.6823 | -16.27 | -27.85 | 126.69 | wait for HOME reversal trigger, then label capitulation rebound |
| SIREN | Siren | chase_risk | wait_or_fade_watch | 48.8490 | 76 | 0.1414 | 0.51 | 116.66 | -5.08 | avoid chasing SIREN; label pullback or fade setup first |
| BCH | Bitcoin Cash | capitulation_reversal_watch | watch_reversal_trigger | 48.3660 | 26 | 0.1300 | -7.04 | -26.86 | -54.34 | wait for BCH reversal trigger, then label capitulation rebound |
| BTW | Bitway | chase_risk | wait_or_fade_watch | 48.3058 | 225 | 0.2426 | -0.82 | 384.32 | 331.89 | avoid chasing BTW; label pullback or fade setup first |
| ZEC | Zcash | volume_reversal_candidate | long_reversal | 48.1265 | 15 | 0.1051 | 5.56 | -22.02 | -29.69 | paper-label ZEC volume-backed reversal over 1h, 4h, 12h, and 24h |
| IP | Story | volume_reversal_candidate | long_reversal | 47.9936 | 254 | 0.2861 | 3.53 | -27.16 | -44.63 | paper-label IP volume-backed reversal over 1h, 4h, 12h, and 24h |
| INJ | Injective | volume_reversal_candidate | long_reversal | 46.9236 | 94 | 0.1825 | 9.81 | -15.86 | 33.01 | paper-label INJ volume-backed reversal over 1h, 4h, 12h, and 24h |
| FARTCOIN | Fartcoin | volume_reversal_candidate | long_reversal | 43.7494 | 245 | 0.1946 | 5.47 | -23.86 | -53.29 | paper-label FARTCOIN volume-backed reversal over 1h, 4h, 12h, and 24h |
| VIRTUAL | Virtuals Protocol | volume_reversal_candidate | long_reversal | 42.8300 | 121 | 0.2017 | 4.15 | -17.63 | -37.58 | paper-label VIRTUAL volume-backed reversal over 1h, 4h, 12h, and 24h |
| NIGHT | Midnight | volume_reversal_candidate | long_reversal | 41.7822 | 101 | 0.1255 | 6.24 | -18.06 | -4.12 | paper-label NIGHT volume-backed reversal over 1h, 4h, 12h, and 24h |
| OP | Optimism | volume_reversal_candidate | long_reversal | 41.6116 | 171 | 0.2648 | 3.62 | -15.65 | -39.99 | paper-label OP volume-backed reversal over 1h, 4h, 12h, and 24h |
| SOL | Solana | volume_reversal_candidate | long_reversal | 41.1550 | 7 | 0.0818 | 3.24 | -18.36 | -29.32 | paper-label SOL volume-backed reversal over 1h, 4h, 12h, and 24h |
| VELVET | Velvet | chase_risk | wait_or_fade_watch | 41.0613 | 211 | 0.1102 | 59.35 | 262.11 | 279.78 | avoid chasing VELVET; label pullback or fade setup first |
| HYPE | Hyperliquid | volume_reversal_candidate | long_reversal | 40.6421 | 10 | 0.0552 | 6.57 | -16.26 | 40.53 | paper-label HYPE volume-backed reversal over 1h, 4h, 12h, and 24h |
| EIGEN | EigenCloud (prev. EigenLayer) | volume_reversal_candidate | long_reversal | 40.3280 | 217 | 0.2394 | 8.41 | -13.41 | -15.41 | paper-label EIGEN volume-backed reversal over 1h, 4h, 12h, and 24h |
| PENGU | Pudgy Penguins | volume_reversal_candidate | long_reversal | 39.6821 | 113 | 0.2802 | 3.29 | -10.23 | -35.21 | paper-label PENGU volume-backed reversal over 1h, 4h, 12h, and 24h |
| ETH | Ethereum | volume_reversal_candidate | long_reversal | 38.8798 | 2 | 0.0784 | 3.76 | -15.52 | -27.73 | paper-label ETH volume-backed reversal over 1h, 4h, 12h, and 24h |
| TRUMP | Official Trump | volume_reversal_candidate | long_reversal | 38.6921 | 116 | 0.1856 | 3.26 | -15.09 | -32.50 | paper-label TRUMP volume-backed reversal over 1h, 4h, 12h, and 24h |
| BAT | Basic Attention | volume_reversal_candidate | long_reversal | 37.5697 | 220 | 0.0838 | 3.74 | -24.80 | -20.02 | paper-label BAT volume-backed reversal over 1h, 4h, 12h, and 24h |
| ZRO | LayerZero | capitulation_reversal_watch | watch_reversal_trigger | 37.4495 | 162 | 0.1268 | -0.76 | -22.94 | -40.03 | wait for ZRO reversal trigger, then label capitulation rebound |
| SKYAI | SkyAI | chase_risk | wait_or_fade_watch | 35.2639 | 143 | 0.2135 | 4.61 | 58.43 | -57.55 | avoid chasing SKYAI; label pullback or fade setup first |
| LUNC | Terra Luna Classic | volume_reversal_candidate | long_reversal | 35.2373 | 119 | 0.0917 | 4.70 | -15.98 | -23.31 | paper-label LUNC volume-backed reversal over 1h, 4h, 12h, and 24h |
| DYDX | dYdX | volume_reversal_candidate | long_reversal | 34.1366 | 240 | 0.0823 | 3.95 | -22.25 | -19.46 | paper-label DYDX volume-backed reversal over 1h, 4h, 12h, and 24h |
| LINK | Chainlink | volume_reversal_candidate | long_reversal | 32.6087 | 20 | 0.0544 | 3.18 | -12.17 | -24.50 | paper-label LINK volume-backed reversal over 1h, 4h, 12h, and 24h |
| IMX | Immutable | volume_reversal_candidate | long_reversal | 31.4084 | 241 | 0.1180 | 9.39 | -11.99 | -26.39 | paper-label IMX volume-backed reversal over 1h, 4h, 12h, and 24h |
| ENS | Ethereum Name Service | volume_reversal_candidate | long_reversal | 31.2864 | 177 | 0.0860 | 3.09 | -16.88 | -30.65 | paper-label ENS volume-backed reversal over 1h, 4h, 12h, and 24h |
| IOTA | IOTA | volume_reversal_candidate | long_reversal | 31.1693 | 170 | 0.0413 | 4.82 | -17.37 | -22.81 | paper-label IOTA volume-backed reversal over 1h, 4h, 12h, and 24h |
| TON | Toncoin | volume_reversal_candidate | long_reversal | 30.3373 | 22 | 0.0470 | 3.26 | -10.35 | -30.33 | paper-label TON volume-backed reversal over 1h, 4h, 12h, and 24h |
| GWEI | ETHGas | chase_risk | wait_or_fade_watch | 28.9449 | 133 | 0.1483 | 10.53 | 46.79 | 12.94 | avoid chasing GWEI; label pullback or fade setup first |

## Interpretation

`volume_reversal_candidate` looks for heavy-volume rebound after a weak 7d move. `capitulation_reversal_watch` is a falling setup that still needs a trigger. `breakout_continuation_watch` is already moving and needs stop/entry discipline. `chase_risk` should usually be avoided until pullback or fade labels exist.
