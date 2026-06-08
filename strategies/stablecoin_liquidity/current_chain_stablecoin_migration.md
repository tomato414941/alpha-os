# Current Chain Stablecoin Migration

This aggregates DeFiLlama stablecoin chain-circulating data into chain-level liquidity migration. It is a capital-flow proxy, not a bridge-fill or trade instruction.

| chain | token | status | supply USD | day change | week change | month change | week % | top asset | score | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| Solana | SOL | paper_chain_stablecoin_inflow_watch | 13099796301 | 454833106 | 930837276 | 898112549 | 0.0765 | USDC | 90.2986 | large stablecoin inflow suggests deployable capital is arriving |
| Polygon | POL | paper_chain_stablecoin_outflow_watch | 3435388497 | -89229106 | -212411978 | -64720329 | -0.0582 | USDC | 63.7020 | large stablecoin outflow suggests capital is leaving |
| Ethereum | ETH | chain_stablecoin_flow_reversal_watch | 147257248778 | -133182923 | -2549343868 | -5834768183 | -0.0170 | USDT | 63.4035 | large weekly stablecoin flow has mixed daily confirmation |
| Base | - | paper_chain_stablecoin_inflow_watch | 4475481887 | 49566481 | 169236523 | -51398790 | 0.0393 | USDC | 60.7974 | large stablecoin inflow suggests deployable capital is arriving |
| Tron | TRX | chain_stablecoin_flow_reversal_watch | 89349190154 | -104168341 | -455311943 | -20026100 | -0.0051 | USDT | 58.7796 | large weekly stablecoin flow has mixed daily confirmation |
| Aptos | - | chain_stablecoin_context_watch | 1131103210 | 39962373 | 99369542 | 50180407 | 0.0963 | USDT | 40.3622 | large chain has a material stablecoin distribution change |
| Arbitrum | ARB | chain_stablecoin_context_watch | 3671598623 | -42745493 | -97793289 | 145459005 | -0.0259 | PYUSD | 28.7501 | large chain has a material stablecoin distribution change |
| Starknet | - | chain_stablecoin_context | 180477312 | 131487 | -81675886 | -135246602 | -0.3116 | USDC | 24.2643 | chain stablecoin distribution context |
| Stellar | - | chain_stablecoin_context | 270505001 | 626604 | 65828329 | -5626340 | 0.3216 | USDC | 23.5619 | chain stablecoin distribution context |
| Ink | - | chain_stablecoin_context | 180320546 | -5930410 | 26150368 | -10779980 | 0.1696 | USDC | 21.4878 | chain stablecoin distribution context |
| XDC | - | chain_stablecoin_context | 86969837 | -411 | 25149357 | 11554237 | 0.4068 | USDC | 21.3444 | chain stablecoin distribution context |
| ZKsync Era | - | chain_stablecoin_context | 35833346 | -3624 | -6793080 | -7143694 | -0.1594 | USDC | 20.3755 | chain stablecoin distribution context |
| Cardano | - | chain_stablecoin_context | 20362510 | -168230 | -6004799 | 3282109 | -0.2277 | USDC | 20.3206 | chain stablecoin distribution context |
| World Chain | - | chain_stablecoin_context | 25419355 | 227096 | 4630178 | 6468127 | 0.2227 | USDC | 20.2569 | chain stablecoin distribution context |
| Flow | - | chain_stablecoin_context | 21427742 | -85544 | 2274597 | 9864990 | 0.1188 | PYUSD | 20.1352 | chain stablecoin distribution context |
| Katana | - | chain_stablecoin_context | 23033662 | -822742 | -2410893 | -51598379 | -0.0948 | USDC | 19.0937 | chain stablecoin distribution context |
| Sui | SUI | chain_stablecoin_context | 359640677 | -1717875 | 25095019 | -83264046 | 0.0750 | USDC | 16.6168 | chain stablecoin distribution context |
| Algorand | - | chain_stablecoin_context | 56076902 | 2323737 | 4029241 | -26837727 | 0.0774 | USDC | 15.7404 | chain stablecoin distribution context |
| Avalanche | AVAX | chain_stablecoin_context | 811726105 | -70542692 | -49109349 | -295202281 | -0.0570 | USDT | 14.6769 | chain stablecoin distribution context |
| Berachain | BERA | chain_stablecoin_context | 70206422 | 180455 | -5189917 | -7417247 | -0.0688 | USDe | 14.0967 | chain stablecoin distribution context |
| BSC | BNB | chain_stablecoin_context | 12545366522 | -612770 | -59677139 | -131362815 | -0.0047 | USD1 | 13.9307 | chain stablecoin distribution context |
| MegaETH | - | chain_stablecoin_context | 438776014 | 3142818 | 16605276 | 33741170 | 0.0393 | USDe | 9.1357 | chain stablecoin distribution context |
| Hyperliquid L1 | HYPE | chain_stablecoin_context | 6584892929 | 192655600 | 28010708 | 1298608353 | 0.0043 | USDC | 8.8398 | chain stablecoin distribution context |
| Osmosis | - | chain_stablecoin_context | 22483968 | 141444 | -533192 | -1080777 | -0.0232 | USDC | 4.6821 | chain stablecoin distribution context |
| Linea | - | chain_stablecoin_context | 34932499 | 16524 | -687223 | -6117820 | -0.0193 | USDC | 3.9280 | chain stablecoin distribution context |

## Interpretation

Stablecoin inflow can indicate deployable capital arriving on a chain; outflow can indicate risk-off, bridge withdrawal, or venue-specific liquidity stress. This still needs token mapping, venue coverage, bridge route checks, and forward labels.
