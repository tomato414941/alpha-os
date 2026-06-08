# Current GeckoTerminal DEX Pool Flow

This screen reads GeckoTerminal trending pools and scores DEX pool activity. It is a pool-flow screen, not a trade instruction.

| network | dex | pool | status | reserve USD | vol 1h | vol/reserve 1h | chg 1h | chg 24h | imbalance 1h | score | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| bsc | pancakeswap-infinity-clmm | Beat / USDT 0.007% | dex_pool_context_watch | 4158991 | 1257397 | 0.3023 | 4.4330 | 72.7550 | -0.0179 | 62.8118 | pool flow is context but not yet actionable |
| bsc | pancakeswap-infinity-clmm | BSB / USDT 0.007% | dex_pool_context_watch | 16995152 | 934765 | 0.0550 | 3.3150 | 22.9910 | -0.0172 | 57.5221 | pool flow is context but not yet actionable |
| base | uniswap-v3-base | WETH / USDC 0.3% | dex_pool_context_watch | 95881405 | 9235667 | 0.0963 | 0.7560 | 3.0940 | 0.1039 | 57.3833 | pool flow is context but not yet actionable |
| solana | raydium | pippin / SOL | dex_pool_context_watch | 3150553 | 321318 | 0.1020 | 0.9410 | 37.7900 | -0.0947 | 57.2446 | pool flow is context but not yet actionable |
| bsc | pancakeswap-infinity-clmm | BTW / USDT 0.007% | dex_pool_context_watch | 1239399 | 264544 | 0.2134 | 6.4970 | -4.1410 | 0.0048 | 51.7323 | pool flow is context but not yet actionable |
| bsc | uniswap-v4-bsc | SIREN / USDT 0.209% | dex_pool_context_watch | 796262 | 216360 | 0.2717 | 6.0580 | 5.3480 | -0.1065 | 43.3356 | pool flow is context but not yet actionable |
| solana | pumpswap | WORLDCUP / SOL | paper_dex_pool_momentum_watch | 342713 | 146035 | 0.4261 | 8.6920 | 46.5080 | 0.2510 | 32.4735 | short-term pool flow and price are aligned |
| solana | pumpswap | TROLL / SOL | dex_pool_context_watch | 2990385 | 113336 | 0.0379 | 3.1170 | 8.0530 | -0.4069 | 31.6904 | pool flow is context but not yet actionable |
| base | uniswap-v3-base | DEGEN / WETH 0.3% | dex_pool_context_watch | 1315613 | 124412 | 0.0946 | 10.6090 | 0.4670 | -0.0928 | 31.1534 | pool flow is context but not yet actionable |
| base | uniswap-v4-base | GITLAWB / WETH | dex_pool_context_watch | 2644992 | 8423 | 0.0032 | 2.3300 | -16.7450 | 0.3846 | 26.6259 | pool flow is context but not yet actionable |
| eth | uniswap_v2 | SPCX / WETH | paper_dex_pool_momentum_watch | 148829 | 42156 | 0.2833 | 11.8250 | 159.1130 | 0.3388 | 25.6484 | short-term pool flow and price are aligned |
| solana | pumpswap | Bountywork / SOL | dex_pool_context_watch | 132854 | 35540 | 0.2675 | -3.5650 | 147.7790 | 0.0845 | 24.0668 | pool flow is context but not yet actionable |
| base | uniswap-v4-base | Surplus / WETH | dex_pool_context_watch | 979579 | 20811 | 0.0212 | 8.4930 | 34.6040 | 0.2195 | 20.7614 | pool flow is context but not yet actionable |
| solana | pumpswap | Magpie / SOL | dex_liquidity_stress_watch | 83304 | 43420 | 0.5212 | -16.9090 | 1.2700 | 0.1626 | 18.9751 | pool turnover is high relative to reserves |
| solana | pumpswap | three / SOL | dex_pool_context_watch | 259297 | 36262 | 0.1398 | -1.6430 | -12.9340 | 0.4223 | 15.8428 | pool flow is context but not yet actionable |
| solana | pumpswap | BOUTYWORK / SOL | dex_microcap_liquidity_watch | 29216 | 32648 | 1.1175 | -28.6980 | -78.2870 | 0.1546 | 12.9471 | pool is too thin for direct action |
| eth | uniswap_v2 | DOGEUS / WETH | dex_pool_context_watch | 234572 | 32600 | 0.1390 | 3.0870 | 19.9970 | 0.1111 | 11.3869 | pool flow is context but not yet actionable |
| base | uniswap-v4-base | Synthetic / WETH | dex_pool_context_watch | 111951 | 3045 | 0.0272 | 2.1000 | -3.0100 | -0.1333 | 1.5001 | pool flow is context but not yet actionable |
| solana | pumpswap | 67 / SOL | dex_pool_context_watch | 317185 | 8434 | 0.0266 | 5.0470 | -4.0500 | -0.8056 | 1.3044 | pool flow is context but not yet actionable |
| solana | pumpswap | $tupid / SOL | dex_microcap_liquidity_watch | 46144 | 23310 | 0.5052 | 2.3400 | -64.1520 | 0.2245 | -3.6664 | pool is too thin for direct action |
