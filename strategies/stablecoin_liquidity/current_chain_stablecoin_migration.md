# Current Chain Stablecoin Migration

This aggregates DeFiLlama stablecoin chain-circulating data into chain-level liquidity migration. It is a capital-flow proxy, not a bridge-fill or trade instruction.

| chain | token | status | supply USD | day change | week change | month change | week % | top asset | score | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| Solana | SOL | paper_chain_stablecoin_inflow_watch | 12843658526 | 329197487 | 679470371 | 559970671 | 0.0559 | USDC | 86.1717 | large stablecoin inflow suggests deployable capital is arriving |
| Ethereum | ETH | chain_stablecoin_flow_reversal_watch | 147440263409 | -167280456 | -2469368033 | -7482131135 | -0.0165 | USDT | 63.2945 | large weekly stablecoin flow has mixed daily confirmation |
| Tron | TRX | chain_stablecoin_flow_reversal_watch | 89460659341 | -291752584 | -564915861 | 589584523 | -0.0063 | USDT | 61.2550 | large weekly stablecoin flow has mixed daily confirmation |
| Base | - | paper_chain_stablecoin_inflow_watch | 4469320291 | 65312298 | 165682202 | -99346502 | 0.0385 | USDC | 60.4531 | large stablecoin inflow suggests deployable capital is arriving |
| Polygon | POL | paper_chain_stablecoin_outflow_watch | 3526763062 | -7255467 | -119439024 | -2398475 | -0.0328 | USDC | 54.0501 | large stablecoin outflow suggests capital is leaving |
| Hyperliquid L1 | HYPE | chain_stablecoin_flow_reversal_watch | 6401175688 | -220544193 | -115862253 | 1147092860 | -0.0178 | USDC | 40.7500 | large weekly stablecoin flow has mixed daily confirmation |
| Aptos | - | chain_stablecoin_context_watch | 1091366590 | 51524949 | 71369796 | -4899471 | 0.0700 | USDT | 33.6540 | large chain has a material stablecoin distribution change |
| Arbitrum | ARB | chain_stablecoin_context_watch | 3741719423 | 31335577 | -64677289 | 166225868 | -0.0170 | PYUSD | 25.3739 | large chain has a material stablecoin distribution change |
| Starknet | - | chain_stablecoin_context | 180427663 | -908989 | -80974075 | -122803646 | -0.3098 | USDC | 24.2291 | chain stablecoin distribution context |
| Stellar | - | chain_stablecoin_context | 271101352 | 1298983 | 65000040 | -2312243 | 0.3154 | USDC | 23.5211 | chain stablecoin distribution context |
| Ink | - | chain_stablecoin_context | 187688515 | -312631 | 34048231 | -6643595 | 0.2216 | USDC | 21.8901 | chain stablecoin distribution context |
| Plasma | - | chain_stablecoin_context | 962062546 | 30247736 | 75846551 | 1902804 | 0.0856 | USDT | 21.8713 | chain stablecoin distribution context |
| XDC | - | chain_stablecoin_context | 86976764 | 27060842 | 25151537 | 21560185 | 0.4068 | USDC | 21.3446 | chain stablecoin distribution context |
| ZKsync Era | - | chain_stablecoin_context | 35836553 | -63609 | -6796590 | -7126349 | -0.1594 | USDC | 20.3757 | chain stablecoin distribution context |
| Katana | - | chain_stablecoin_context | 22195652 | -1888119 | -6994961 | -49446648 | -0.2396 | USDT | 20.3719 | chain stablecoin distribution context |
| Cardano | - | chain_stablecoin_context | 20500184 | -139836 | -5887496 | 3445072 | -0.2231 | USDC | 20.3149 | chain stablecoin distribution context |
| World Chain | - | chain_stablecoin_context | 25305220 | -122029 | 4228053 | 6943066 | 0.2006 | USDC | 20.2367 | chain stablecoin distribution context |
| Flow | - | chain_stablecoin_context | 21458929 | -79065 | 3195031 | 9999901 | 0.1749 | PYUSD | 20.1812 | chain stablecoin distribution context |
| Sui | SUI | chain_stablecoin_context | 362445292 | 1919232 | 28215133 | -73466015 | 0.0844 | USDC | 18.6569 | chain stablecoin distribution context |
| Avalanche | AVAX | chain_stablecoin_context | 819957888 | -82930791 | -55977722 | -266488599 | -0.0639 | USDT | 16.4001 | chain stablecoin distribution context |
| BSC | BNB | chain_stablecoin_context | 12549175820 | -146674 | -66937333 | -134266099 | -0.0053 | USD1 | 14.4080 | chain stablecoin distribution context |
| Algorand | - | chain_stablecoin_context | 55183610 | 1401972 | 3124861 | -27468785 | 0.0600 | USDC | 12.2166 | chain stablecoin distribution context |
| Flare | - | chain_stablecoin_context | 27436504 | 154139 | -1483560 | -175100 | -0.0513 | USDT | 10.3613 | chain stablecoin distribution context |
| Berachain | BERA | chain_stablecoin_context | 70115629 | -6425 | -3191937 | -9197929 | -0.0435 | USDe | 8.9381 | chain stablecoin distribution context |
| MegaETH | - | chain_stablecoin_context | 437649927 | 3045541 | 15783261 | 94374714 | 0.0374 | USDe | 8.7094 | chain stablecoin distribution context |

## Interpretation

Stablecoin inflow can indicate deployable capital arriving on a chain; outflow can indicate risk-off, bridge withdrawal, or venue-specific liquidity stress. This still needs token mapping, venue coverage, bridge route checks, and forward labels.
