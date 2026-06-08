# Current GeckoTerminal DEX Pool Flow

This screen reads GeckoTerminal trending pools and scores DEX pool activity. It is a pool-flow screen, not a trade instruction.

| network | dex | pool | status | reserve USD | vol 1h | vol/reserve 1h | chg 1h | chg 24h | imbalance 1h | score | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| solana | raydium | pippin / SOL | dex_pool_context_watch | 3740128 | 789046 | 0.2110 | 0.8720 | 82.8450 | -0.1079 | 60.4346 | pool flow is context but not yet actionable |
| bsc | pancakeswap-infinity-clmm | Beat / USDT 0.007% | dex_pool_context_watch | 4126193 | 323588 | 0.0784 | -0.2010 | 34.7730 | -0.0292 | 57.2916 | pool flow is context but not yet actionable |
| solana | pumpswap | GO / SOL | paper_dex_reversal_risk_watch | 63504 | 123645 | 1.9470 | 17.1780 | 2511.2810 | 0.0765 | 52.3842 | extreme pool turnover after large 24h move |
| bsc | pancakeswap-v3-bsc | ESPORTS / WBNB 0.01% | dex_pool_context_watch | 1318056 | 447471 | 0.3395 | 1.5380 | -3.2190 | 0.0062 | 52.2594 | pool flow is context but not yet actionable |
| solana | pumpswap | Bountywork / SOL | dex_liquidity_stress_watch | 91471 | 139462 | 1.5247 | -20.2110 | -51.3590 | 0.2939 | 47.6220 | pool turnover is high relative to reserves |
| bsc | uniswap-v4-bsc | SIREN / USDT 0.209% | dex_pool_context_watch | 786910 | 350529 | 0.4455 | 4.4420 | -3.1950 | -0.1287 | 47.5288 | pool flow is context but not yet actionable |
| base | aerodrome-slipstream | VELVET / USDC 0.01% | dex_pool_context_watch | 492148 | 227879 | 0.4630 | 3.7280 | 36.3070 | -0.0090 | 43.4514 | pool flow is context but not yet actionable |
| solana | pumpswap | PARQ / USDC | paper_dex_reversal_risk_watch | 125175 | 98605 | 0.7877 | -13.2740 | 446.5560 | 0.1833 | 39.0642 | extreme pool turnover after large 24h move |
| solana | pumpswap | WORLDCUP / SOL | dex_liquidity_stress_watch | 366551 | 187800 | 0.5123 | 1.0480 | 40.3050 | 0.2271 | 36.6951 | pool turnover is high relative to reserves |
| solana | pumpswap | LIFE / SOL | dex_microcap_liquidity_watch | 27754 | 87626 | 3.1572 | -24.9110 | 361.0730 | 0.0602 | 36.6929 | pool is too thin for direct action |
| solana | raydium-clmm | CARDS / USDC | dex_pool_context_watch | 2894607 | 80270 | 0.0277 | 1.3310 | 8.1350 | 0.3030 | 35.7961 | pool flow is context but not yet actionable |
| solana | pumpswap | three / SOL | dex_pool_context_watch | 251074 | 49724 | 0.1980 | -7.9980 | -21.8350 | 0.4357 | 20.0879 | pool flow is context but not yet actionable |
| base | uniswap-v4-base | GITLAWB / WETH | dex_pool_context_watch | 2799267 | 11323 | 0.0040 | -1.6150 | 1.4640 | -0.2500 | 19.6383 | pool flow is context but not yet actionable |
| base | uniswap-v4-base | Surplus / WETH | dex_pool_context_watch | 899980 | 5210 | 0.0058 | -1.5170 | 31.9650 | 0.4286 | 17.3202 | pool flow is context but not yet actionable |
| eth | uniswap_v2 | SPCX / WETH | dex_pool_context_watch | 174310 | 19549 | 0.1122 | 8.1060 | 40.6100 | 0.5161 | 16.4243 | pool flow is context but not yet actionable |
| base | uniswap-v4-base | rootai / WETH | dex_pool_context_watch | 337392 | 3134 | 0.0093 | -1.2070 | 15.1070 | 0.5238 | 10.4032 | pool flow is context but not yet actionable |
| eth | uniswap_v2 | DOGEUS / WETH | dex_pool_context_watch | 226105 | 20055 | 0.0887 | -3.8040 | -2.9790 | 0.2000 | 9.8491 | pool flow is context but not yet actionable |
| solana | pumpswap | 67 / SOL | dex_pool_context_watch | 314298 | 3340 | 0.0106 | -1.7600 | 4.6630 | 0.3333 | 7.8769 | pool flow is context but not yet actionable |
| base | uniswap-v4-base | Synthetic / WETH | dex_pool_context_watch | 89064 | 5542 | 0.0622 | -4.4360 | -39.6280 | -0.4737 | 0.9282 | pool flow is context but not yet actionable |
| base | uniswap-v3-base | BLKH / WETH 1% | dex_microcap_liquidity_watch | 4952 | 34 | 0.0068 | 3.4400 | -19.3170 | 1.0000 | -7.8798 | pool is too thin for direct action |
