# Current GeckoTerminal DEX Pool Flow

This screen reads GeckoTerminal trending pools and scores DEX pool activity. It is a pool-flow screen, not a trade instruction.

| network | dex | pool | status | reserve USD | vol 1h | vol/reserve 1h | chg 1h | chg 24h | imbalance 1h | score | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| bsc | pancakeswap-infinity-clmm | Beat / USDT 0.007% | dex_pool_context_watch | 4094353 | 1050501 | 0.2566 | 1.4100 | 60.1150 | -0.0359 | 60.6357 | pool flow is context but not yet actionable |
| solana | raydium | pippin / SOL | dex_pool_context_watch | 3196001 | 297785 | 0.0932 | 0.2570 | 38.7570 | 0.0219 | 58.1653 | pool flow is context but not yet actionable |
| base | uniswap-v3-base | WETH / USDC 0.3% | dex_pool_context_watch | 96364195 | 3280070 | 0.0340 | 0.5700 | 3.9420 | 0.2438 | 58.1466 | pool flow is context but not yet actionable |
| bsc | pancakeswap-infinity-clmm | BSB / USDT 0.007% | dex_pool_context_watch | 16935145 | 619206 | 0.0366 | 6.0710 | 1.6240 | 0.0164 | 57.4326 | pool flow is context but not yet actionable |
| solana | pumpswap | Bountywork / SOL | dex_liquidity_stress_watch | 110935 | 177836 | 1.6031 | -19.0990 | 97.2470 | 0.1811 | 54.9342 | pool turnover is high relative to reserves |
| solana | pumpswap | TROLL / SOL | dex_pool_context_watch | 3122843 | 418438 | 0.1340 | 4.1060 | 15.4010 | 0.0377 | 52.4120 | pool flow is context but not yet actionable |
| solana | raydium-clmm | CARDS / USDC | dex_pool_context_watch | 2934012 | 160831 | 0.0548 | -3.9730 | -6.2500 | 0.0732 | 42.9686 | pool flow is context but not yet actionable |
| bsc | uniswap-v4-bsc | SIREN / USDT 0.209% | dex_pool_context_watch | 761093 | 166046 | 0.2182 | -1.2580 | -9.3820 | -0.0271 | 36.9723 | pool flow is context but not yet actionable |
| solana | pumpswap | WORLDCUP / SOL | paper_dex_pool_momentum_watch | 350326 | 113223 | 0.3232 | 9.1450 | 49.7970 | 0.1380 | 28.4588 | short-term pool flow and price are aligned |
| solana | pumpswap | three / SOL | dex_pool_context_watch | 261291 | 60038 | 0.2298 | -8.2900 | -22.6880 | 0.4265 | 21.5479 | pool flow is context but not yet actionable |
| eth | uniswap_v2 | SPCX / WETH | dex_pool_context_watch | 172176 | 26934 | 0.1564 | 17.8820 | 10.1170 | 0.6623 | 20.5324 | pool flow is context but not yet actionable |
| base | uniswap-v3-base | DEGEN / WETH 0.3% | dex_pool_context_watch | 1253141 | 31675 | 0.0253 | 0.2040 | -9.3130 | -0.0554 | 18.3041 | pool flow is context but not yet actionable |
| base | uniswap-v4-base | Surplus / WETH | dex_pool_context_watch | 985823 | 8360 | 0.0085 | 1.9610 | 28.1570 | 0.3500 | 17.9626 | pool flow is context but not yet actionable |
| base | uniswap-v4-base | GITLAWB / WETH | dex_pool_context_watch | 2766109 | 8576 | 0.0031 | -2.9990 | -9.5520 | -0.8000 | 17.6660 | pool flow is context but not yet actionable |
| eth | uniswap_v2 | DOGEUS / WETH | dex_pool_context_watch | 227988 | 50503 | 0.2215 | -3.2820 | 3.1930 | 0.1579 | 13.6602 | pool flow is context but not yet actionable |
| solana | pumpswap | Magpie / SOL | dex_pool_context_watch | 84987 | 30086 | 0.3540 | 2.2840 | 35.7430 | 0.0331 | 11.7070 | pool flow is context but not yet actionable |
| solana | pumpswap | 67 / SOL | dex_pool_context_watch | 315881 | 1559 | 0.0049 | 1.1260 | -4.9650 | 0.1429 | 5.6596 | pool flow is context but not yet actionable |
| base | uniswap-v4-base | Synthetic / WETH | dex_pool_context_watch | 113229 | 7298 | 0.0645 | -5.9190 | -15.1430 | -0.2195 | 3.1330 | pool flow is context but not yet actionable |
| solana | pumpswap | BOUTYWORK / SOL | dex_microcap_liquidity_watch | 26754 | 21174 | 0.7914 | 11.7180 | -78.8370 | 0.0000 | 1.2410 | pool is too thin for direct action |
| base | uniswap-v4-base | rootai / WETH | dex_pool_context_watch | 335699 | 2588 | 0.0077 | -2.2800 | 9.6880 | -0.7778 | 0.1184 | pool flow is context but not yet actionable |
