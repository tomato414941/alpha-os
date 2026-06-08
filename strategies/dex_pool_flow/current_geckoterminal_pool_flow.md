# Current GeckoTerminal DEX Pool Flow

This screen reads GeckoTerminal trending pools and scores DEX pool activity. It is a pool-flow screen, not a trade instruction.

| network | dex | pool | status | reserve USD | vol 1h | vol/reserve 1h | chg 1h | chg 24h | imbalance 1h | score | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| bsc | pancakeswap-infinity-clmm | Beat / USDT 0.007% | dex_pool_context_watch | 4221729 | 1394149 | 0.3302 | 4.6280 | 74.8230 | -0.0310 | 63.1219 | pool flow is context but not yet actionable |
| solana | raydium | pippin / SOL | dex_pool_context_watch | 3150302 | 339836 | 0.1079 | 3.2250 | 43.3220 | 0.0041 | 59.2530 | pool flow is context but not yet actionable |
| bsc | pancakeswap-infinity-clmm | BSB / USDT 0.007% | dex_pool_context_watch | 17334816 | 901209 | 0.0520 | 4.5710 | 28.5550 | -0.0143 | 58.1762 | pool flow is context but not yet actionable |
| base | uniswap-v3-base | WETH / USDC 0.3% | dex_pool_context_watch | 95822118 | 7714888 | 0.0805 | -0.5830 | 3.8120 | -0.1824 | 54.3462 | pool flow is context but not yet actionable |
| bsc | pancakeswap-infinity-clmm | BTW / USDT 0.007% | dex_pool_context_watch | 1241468 | 306170 | 0.2466 | 6.2060 | 1.9250 | 0.0111 | 51.9496 | pool flow is context but not yet actionable |
| bsc | uniswap-v4-bsc | SIREN / USDT 0.209% | dex_pool_context_watch | 774133 | 238015 | 0.3075 | -2.7080 | 1.8250 | -0.0695 | 44.8265 | pool flow is context but not yet actionable |
| solana | pumpswap | WORLDCUP / SOL | paper_dex_pool_momentum_watch | 343499 | 208525 | 0.6071 | 8.6010 | 51.1050 | 0.1818 | 40.2060 | short-term pool flow and price are aligned |
| solana | pumpswap | TROLL / SOL | dex_pool_context_watch | 2958310 | 133976 | 0.0453 | -0.1960 | 6.8930 | -0.2865 | 34.1245 | pool flow is context but not yet actionable |
| eth | uniswap_v2 | SPCX / WETH | paper_dex_pool_momentum_watch | 147611 | 46740 | 0.3166 | 7.5210 | 196.7940 | 0.3594 | 27.2339 | short-term pool flow and price are aligned |
| base | uniswap-v4-base | GITLAWB / WETH | dex_pool_context_watch | 2629316 | 7547 | 0.0029 | 3.8180 | -14.3170 | 0.3846 | 26.8582 | pool flow is context but not yet actionable |
| solana | pumpswap | Bountywork / SOL | dex_pool_context_watch | 128534 | 31322 | 0.2437 | -8.1380 | 155.4150 | 0.0914 | 25.1786 | pool flow is context but not yet actionable |
| base | uniswap-v3-base | DEGEN / WETH 0.3% | dex_pool_context_watch | 1275765 | 111155 | 0.0871 | 0.7170 | -3.5620 | -0.3032 | 24.5149 | pool flow is context but not yet actionable |
| solana | pumpswap | three / SOL | paper_dex_pool_momentum_watch | 267145 | 51589 | 0.1931 | 5.0890 | -0.5180 | 0.3956 | 18.1531 | short-term pool flow and price are aligned |
| base | uniswap-v4-base | Surplus / WETH | dex_pool_context_watch | 980722 | 6239 | 0.0064 | -1.6990 | 37.1960 | 0.1613 | 16.6501 | pool flow is context but not yet actionable |
| solana | pumpswap | Magpie / SOL | dex_pool_context_watch | 82323 | 39630 | 0.4814 | -6.1510 | 8.5440 | 0.0794 | 14.4755 | pool flow is context but not yet actionable |
| solana | pumpswap | BOUTYWORK / SOL | dex_microcap_liquidity_watch | 30004 | 43865 | 1.4620 | -19.3070 | -80.3300 | 0.0923 | 14.0881 | pool is too thin for direct action |
| eth | uniswap_v2 | DOGEUS / WETH | dex_pool_context_watch | 232084 | 43427 | 0.1871 | -0.4750 | 14.7110 | 0.1607 | 12.3979 | pool flow is context but not yet actionable |
| solana | pumpswap | 67 / SOL | dex_pool_context_watch | 315099 | 6263 | 0.0199 | 3.5500 | -4.7440 | -0.8182 | 0.5852 | pool flow is context but not yet actionable |
| base | uniswap-v4-base | Synthetic / WETH | dex_pool_context_watch | 111239 | 1644 | 0.0148 | -4.9630 | -7.7320 | -0.3636 | 0.0177 | pool flow is context but not yet actionable |
| solana | pumpswap | $tupid / SOL | dex_microcap_liquidity_watch | 47512 | 32980 | 0.6941 | 1.9150 | -63.3570 | 0.1404 | -1.7889 | pool is too thin for direct action |
