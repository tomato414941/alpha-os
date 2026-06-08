# Current GeckoTerminal DEX Pool Flow

This screen reads GeckoTerminal trending pools and scores DEX pool activity. It is a pool-flow screen, not a trade instruction.

| network | dex | pool | status | reserve USD | vol 1h | vol/reserve 1h | chg 1h | chg 24h | imbalance 1h | score | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| bsc | pancakeswap-infinity-clmm | Beat / USDT 0.007% | dex_pool_context_watch | 4058060 | 1503289 | 0.3704 | -3.7060 | 70.1420 | -0.0333 | 62.9907 | pool flow is context but not yet actionable |
| solana | raydium | pippin / SOL | dex_pool_context_watch | 3116692 | 363580 | 0.1167 | -2.2720 | 39.8880 | -0.0858 | 57.9846 | pool flow is context but not yet actionable |
| bsc | pancakeswap-infinity-clmm | BSB / USDT 0.007% | dex_pool_context_watch | 16643724 | 975219 | 0.0586 | -4.3860 | 18.9210 | -0.0235 | 57.6130 | pool flow is context but not yet actionable |
| base | uniswap-v3-base | WETH / USDC 0.3% | dex_pool_context_watch | 95753998 | 7110328 | 0.0743 | -0.6580 | 4.0410 | -0.2830 | 53.3118 | pool flow is context but not yet actionable |
| bsc | pancakeswap-infinity-clmm | BTW / USDT 0.007% | dex_pool_context_watch | 1239947 | 308818 | 0.2491 | 6.6600 | 1.4580 | 0.0152 | 52.1126 | pool flow is context but not yet actionable |
| solana | pumpswap | WORLDCUP / SOL | paper_dex_pool_momentum_watch | 355795 | 268934 | 0.7559 | 13.8570 | 58.0760 | 0.1626 | 47.8412 | short-term pool flow and price are aligned |
| bsc | uniswap-v4-bsc | SIREN / USDT 0.209% | dex_pool_context_watch | 759236 | 231065 | 0.3043 | -4.5730 | -3.8040 | -0.0268 | 45.0361 | pool flow is context but not yet actionable |
| solana | pumpswap | TROLL / SOL | dex_pool_context_watch | 2972327 | 121322 | 0.0408 | 0.2670 | 7.6100 | -0.2125 | 33.6302 | pool flow is context but not yet actionable |
| eth | uniswap_v2 | SPCX / WETH | paper_dex_pool_momentum_watch | 152596 | 46306 | 0.3035 | 12.8520 | 228.6430 | 0.4351 | 29.6154 | short-term pool flow and price are aligned |
| solana | pumpswap | Bountywork / SOL | dex_pool_context_watch | 124557 | 37819 | 0.3036 | -15.7650 | 134.6080 | 0.1372 | 28.0956 | pool flow is context but not yet actionable |
| base | uniswap-v4-base | GITLAWB / WETH | dex_pool_context_watch | 2683971 | 4315 | 0.0016 | 0.8860 | -14.0030 | 0.4545 | 26.3258 | pool flow is context but not yet actionable |
| base | uniswap-v3-base | DEGEN / WETH 0.3% | dex_pool_context_watch | 1277431 | 100932 | 0.0790 | -1.1600 | -3.0200 | -0.4373 | 22.1932 | pool flow is context but not yet actionable |
| solana | pumpswap | three / SOL | paper_dex_pool_momentum_watch | 269866 | 64501 | 0.2390 | 6.1330 | -1.1510 | 0.3800 | 20.1365 | short-term pool flow and price are aligned |
| base | uniswap-v4-base | Surplus / WETH | dex_pool_context_watch | 992489 | 9222 | 0.0093 | 1.3670 | 40.2110 | 0.2143 | 17.6667 | pool flow is context but not yet actionable |
| eth | uniswap_v2 | DOGEUS / WETH | dex_pool_context_watch | 232205 | 43565 | 0.1876 | -1.2410 | 15.2040 | 0.2364 | 13.4230 | pool flow is context but not yet actionable |
| solana | pumpswap | Magpie / SOL | dex_pool_context_watch | 84054 | 40023 | 0.4762 | -3.4520 | 6.6180 | 0.0590 | 13.3671 | pool flow is context but not yet actionable |
| solana | pumpswap | BOUTYWORK / SOL | dex_microcap_liquidity_watch | 31534 | 44193 | 1.4014 | -5.8220 | -75.4290 | 0.0314 | 8.6048 | pool is too thin for direct action |
| solana | pumpswap | 67 / SOL | dex_pool_context_watch | 315422 | 6527 | 0.0207 | 3.0280 | -4.7280 | -0.8154 | 0.4659 | pool flow is context but not yet actionable |
| base | uniswap-v4-base | Synthetic / WETH | dex_pool_context_watch | 111356 | 979 | 0.0088 | -4.9540 | -12.3360 | -0.5294 | -1.2461 | pool flow is context but not yet actionable |
| solana | pumpswap | $tupid / SOL | dex_microcap_liquidity_watch | 47589 | 34535 | 0.7257 | -1.2380 | -63.8680 | 0.1497 | -1.3986 | pool is too thin for direct action |
