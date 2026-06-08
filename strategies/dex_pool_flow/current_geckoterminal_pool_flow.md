# Current GeckoTerminal DEX Pool Flow

This screen reads GeckoTerminal trending pools and scores DEX pool activity. It is a pool-flow screen, not a trade instruction.

| network | dex | pool | status | reserve USD | vol 1h | vol/reserve 1h | chg 1h | chg 24h | imbalance 1h | score | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| bsc | pancakeswap-infinity-clmm | Beat / USDT 0.007% | dex_pool_context_watch | 4149070 | 959546 | 0.2313 | 7.5350 | 60.3530 | -0.0349 | 62.2414 | pool flow is context but not yet actionable |
| base | uniswap-v3-base | WETH / USDC 0.3% | dex_pool_context_watch | 96510436 | 3674703 | 0.0381 | 0.6980 | 4.4880 | 0.5432 | 61.2467 | pool flow is context but not yet actionable |
| solana | raydium | pippin / SOL | dex_pool_context_watch | 3207072 | 306866 | 0.0957 | 1.2520 | 39.4810 | 0.0340 | 58.6469 | pool flow is context but not yet actionable |
| bsc | pancakeswap-infinity-clmm | BSB / USDT 0.007% | dex_pool_context_watch | 17242115 | 725469 | 0.0421 | 7.9780 | 1.1890 | 0.0227 | 58.1002 | pool flow is context but not yet actionable |
| solana | pumpswap | TROLL / SOL | dex_pool_context_watch | 3093550 | 340539 | 0.1101 | -1.9980 | 13.4950 | 0.0895 | 52.0196 | pool flow is context but not yet actionable |
| base | aerodrome-slipstream | VELVET / USDC 0.01% | dex_pool_context_watch | 522927 | 239512 | 0.4580 | 6.3760 | 72.5720 | 0.0041 | 46.5299 | pool flow is context but not yet actionable |
| solana | pumpswap | Bountywork / SOL | paper_dex_reversal_risk_watch | 113153 | 149343 | 1.3198 | 2.8920 | 117.6700 | 0.0367 | 44.0281 | extreme pool turnover after large 24h move |
| solana | raydium-clmm | CARDS / USDC | dex_pool_context_watch | 2932977 | 173322 | 0.0591 | -3.2360 | -5.3280 | 0.0640 | 43.9420 | pool flow is context but not yet actionable |
| bsc | uniswap-v4-bsc | SIREN / USDT 0.209% | dex_pool_context_watch | 754999 | 150673 | 0.1996 | -3.6400 | -10.9000 | -0.0259 | 35.9905 | pool flow is context but not yet actionable |
| solana | pumpswap | WORLDCUP / SOL | paper_dex_pool_momentum_watch | 352176 | 105325 | 0.2991 | 7.2000 | 51.1570 | 0.1488 | 27.0622 | short-term pool flow and price are aligned |
| solana | pumpswap | three / SOL | dex_pool_context_watch | 260900 | 85517 | 0.3278 | -6.6700 | -30.2680 | 0.4366 | 25.0368 | pool flow is context but not yet actionable |
| base | uniswap-v3-base | DEGEN / WETH 0.3% | dex_pool_context_watch | 1258301 | 39233 | 0.0312 | 0.3270 | -7.7720 | 0.1085 | 20.7838 | pool flow is context but not yet actionable |
| eth | uniswap_v2 | SPCX / WETH | dex_pool_context_watch | 169382 | 27087 | 0.1599 | 10.5870 | 13.7030 | 0.6164 | 18.0812 | pool flow is context but not yet actionable |
| base | uniswap-v4-base | Surplus / WETH | dex_pool_context_watch | 983481 | 11194 | 0.0114 | 1.3680 | 30.4930 | 0.3333 | 18.0176 | pool flow is context but not yet actionable |
| eth | uniswap_v2 | DOGEUS / WETH | dex_pool_context_watch | 228123 | 52931 | 0.2320 | -5.3230 | -0.0720 | 0.1447 | 14.3297 | pool flow is context but not yet actionable |
| solana | pumpswap | Magpie / SOL | dex_pool_context_watch | 84597 | 29322 | 0.3466 | 6.2060 | 30.0400 | -0.0305 | 11.7957 | pool flow is context but not yet actionable |
| solana | pumpswap | 67 / SOL | dex_pool_context_watch | 318882 | 1751 | 0.0055 | 1.6300 | -3.9170 | 0.4286 | 8.6710 | pool flow is context but not yet actionable |
| base | uniswap-v4-base | Synthetic / WETH | dex_pool_context_watch | 113866 | 6234 | 0.0547 | -11.2930 | -15.1710 | -0.3514 | 3.2221 | pool flow is context but not yet actionable |
| solana | pumpswap | BOUTYWORK / SOL | dex_microcap_liquidity_watch | 27572 | 22975 | 0.8333 | 14.7640 | -75.9050 | -0.0580 | 1.9716 | pool is too thin for direct action |
| base | uniswap-v4-base | rootai / WETH | dex_pool_context_watch | 336395 | 2744 | 0.0082 | -3.8190 | 9.6820 | -0.6923 | 0.6071 | pool flow is context but not yet actionable |
