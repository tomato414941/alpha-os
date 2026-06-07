# Current GeckoTerminal DEX Pool Flow

This screen reads GeckoTerminal trending pools and scores DEX pool activity. It is a pool-flow screen, not a trade instruction.

| network | dex | pool | status | reserve USD | vol 1h | vol/reserve 1h | chg 1h | chg 24h | imbalance 1h | score | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| solana | pumpswap | Bountywork / SOL | paper_dex_reversal_risk_watch | 130684 | 257623 | 1.9713 | -11.5720 | 823.6320 | 0.1276 | 68.5613 | extreme pool turnover after large 24h move |
| solana | pumpswap | BOUTYWORK / SOL | paper_dex_reversal_risk_watch | 53447 | 118648 | 2.2199 | -34.5950 | 1438.7750 | 0.1116 | 65.2123 | extreme pool turnover after large 24h move |
| base | uniswap-v3-base | WETH / USDC 0.3% | dex_pool_context_watch | 93776836 | 2950778 | 0.0315 | 0.5250 | 4.4560 | 0.5507 | 61.2022 | pool flow is context but not yet actionable |
| bsc | pancakeswap-infinity-clmm | BSB / USDT 0.007% | dex_pool_context_watch | 18113957 | 736034 | 0.0406 | 5.2020 | 70.1800 | -0.0313 | 60.1626 | pool flow is context but not yet actionable |
| solana | pumpswap | YETZY / SOL | dex_pool_context_watch | 106946 | 24883 | 0.2327 | 5.5710 | 8070.9490 | 0.6018 | 33.5737 | pool flow is context but not yet actionable |
| base | uniswap-v4-base | GITLAWB / WETH | dex_pool_context_watch | 2580678 | 61387 | 0.0238 | -9.9510 | -17.0280 | 0.1765 | 32.4938 | pool flow is context but not yet actionable |
| base | uniswap-v3-base | DEGEN / WETH 0.3% | dex_pool_context_watch | 1300267 | 80425 | 0.0619 | 6.1510 | 7.4040 | -0.0973 | 26.7159 | pool flow is context but not yet actionable |
| solana | pumpswap | $tupid / SOL | paper_dex_pool_momentum_watch | 55000 | 52675 | 0.9577 | 9.8420 | -61.3920 | 0.1208 | 26.4517 | short-term pool flow and price are aligned |
| solana | pumpswap | TROLL / SOL | dex_pool_context_watch | 2801476 | 26776 | 0.0096 | -0.0270 | -0.9760 | -0.1053 | 23.9652 | pool flow is context but not yet actionable |
| base | uniswap-v4-base | Surplus / WETH | dex_pool_context_watch | 811780 | 24663 | 0.0304 | 2.5360 | -45.9950 | 0.0189 | 20.7468 | pool flow is context but not yet actionable |
| eth | uniswap_v2 | SPCX / WETH | dex_pool_context_watch | 136344 | 33168 | 0.2433 | -1.5590 | 13.6260 | 0.1860 | 12.4275 | pool flow is context but not yet actionable |
| solana | pumpswap | three / SOL | dex_pool_context_watch | 280495 | 28852 | 0.1029 | 4.3880 | 23.1660 | -0.0506 | 11.8792 | pool flow is context but not yet actionable |
| base | uniswap-v4-base | Synthetic / WETH | dex_pool_context_watch | 115810 | 4233 | 0.0365 | 9.5840 | 126.5550 | -0.0476 | 11.3880 | pool flow is context but not yet actionable |
| solana | pumpswap | Magpie / SOL | dex_pool_context_watch | 74847 | 14386 | 0.1922 | -9.1150 | -6.5720 | 0.1237 | 11.2344 | pool flow is context but not yet actionable |
| solana | pumpswap | WORLDCUP / SOL | dex_pool_context_watch | 292903 | 27620 | 0.0943 | -0.2280 | 8.1080 | 0.2243 | 11.1221 | pool flow is context but not yet actionable |
| solana | pumpswap | Worlds / SOL | dex_microcap_liquidity_watch | 33345 | 27498 | 0.8247 | 7.2390 | 761.3120 | 0.0357 | 5.4214 | pool is too thin for direct action |
| base | uniswap-v2-base | OS / VIRTUAL | dex_pool_context_watch | 64912 | 1688 | 0.0260 | -2.2340 | 75.6750 | -0.1429 | 4.6011 | pool flow is context but not yet actionable |
| solana | pumpswap | 67 / SOL | dex_pool_context_watch | 298424 | 13137 | 0.0440 | 3.2230 | -5.0650 | -0.5169 | 1.2438 | pool flow is context but not yet actionable |
| solana | pumpswap | + / SOL | dex_microcap_liquidity_watch | 19424 | 11474 | 0.5907 | -3.6160 | -42.4350 | 0.0719 | -7.3823 | pool is too thin for direct action |
| base | uniswap-v3-base | BLKH / WETH 1% | dex_microcap_liquidity_watch | 4188 | 3 | 0.0006 | -2.0000 | 15.2620 | 0.0000 | -18.5883 | pool is too thin for direct action |
