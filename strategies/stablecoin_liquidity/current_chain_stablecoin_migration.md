# Current Chain Stablecoin Migration

This aggregates DeFiLlama stablecoin chain-circulating data into chain-level liquidity migration. It is a capital-flow proxy, not a bridge-fill or trade instruction.

| chain | token | status | supply USD | day change | week change | month change | week % | top asset | score | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| Solana | SOL | paper_chain_stablecoin_inflow_watch | 12683238481 | 37134194 | 513213636 | 480500785 | 0.0422 | USDC | 83.4341 | large stablecoin inflow suggests deployable capital is arriving |
| Ethereum | ETH | chain_stablecoin_flow_reversal_watch | 147576892431 | 183251918 | -2233051618 | -5518824663 | -0.0149 | USDT | 62.9812 | large weekly stablecoin flow has mixed daily confirmation |
| Polygon | POL | paper_chain_stablecoin_outflow_watch | 3456642720 | -68136658 | -191329146 | -43623015 | -0.0524 | USDC | 61.5127 | large stablecoin outflow suggests capital is leaving |
| Base | - | paper_chain_stablecoin_inflow_watch | 4468686072 | 42435357 | 162114788 | -58537962 | 0.0376 | USDC | 60.1031 | large stablecoin inflow suggests deployable capital is arriving |
| Tron | TRX | chain_stablecoin_flow_reversal_watch | 89342194073 | -109214599 | -460358263 | -25081503 | -0.0051 | USDT | 59.0432 | large weekly stablecoin flow has mixed daily confirmation |
| Aptos | - | chain_stablecoin_context_watch | 1131502154 | 40372013 | 99775522 | 50576445 | 0.0967 | USDT | 40.4617 | large chain has a material stablecoin distribution change |
| Hyperliquid L1 | HYPE | chain_stablecoin_context_watch | 6460364403 | 67649200 | -97009456 | 1173686890 | -0.0148 | USDC | 29.2696 | large chain has a material stablecoin distribution change |
| Arbitrum | ARB | chain_stablecoin_context_watch | 3698423965 | -16196168 | -71285869 | 172005234 | -0.0189 | PYUSD | 26.0448 | large chain has a material stablecoin distribution change |
| Starknet | - | chain_stablecoin_context | 180506661 | 147062 | -81666725 | -135241674 | -0.3115 | USDC | 24.2638 | chain stablecoin distribution context |
| Stellar | - | chain_stablecoin_context | 271256828 | 1357255 | 66564096 | -4896179 | 0.3252 | USDC | 23.5995 | chain stablecoin distribution context |
| Ink | - | chain_stablecoin_context | 186809075 | 554306 | 32638518 | -4292801 | 0.2117 | USDC | 21.8187 | chain stablecoin distribution context |
| ZKsync Era | - | chain_stablecoin_context | 35835848 | -3624 | -6793619 | -7144272 | -0.1594 | USDC | 20.3755 | chain stablecoin distribution context |
| Cardano | - | chain_stablecoin_context | 20365614 | -166706 | -6003730 | 3283909 | -0.2277 | USDC | 20.3206 | chain stablecoin distribution context |
| World Chain | - | chain_stablecoin_context | 25435549 | 241313 | 4644741 | 6482833 | 0.2234 | USDC | 20.2577 | chain stablecoin distribution context |
| Katana | - | chain_stablecoin_context | 22721447 | -1135989 | -2724208 | -51913108 | -0.1071 | USDC | 20.1589 | chain stablecoin distribution context |
| Flow | - | chain_stablecoin_context | 21427986 | -94958 | 2266378 | 9860592 | 0.1183 | PYUSD | 20.1347 | chain stablecoin distribution context |
| Sui | SUI | chain_stablecoin_context | 361305034 | -64815 | 26750349 | -81617305 | 0.0800 | USDC | 17.6904 | chain stablecoin distribution context |
| Avalanche | AVAX | chain_stablecoin_context | 807393984 | -74898767 | -53463744 | -299568821 | -0.0621 | USDT | 15.9016 | chain stablecoin distribution context |
| Berachain | BERA | chain_stablecoin_context | 70025297 | -1149 | -5371791 | -7599029 | -0.0712 | USDe | 14.5879 | chain stablecoin distribution context |
| Algorand | - | chain_stablecoin_context | 55781202 | 2023905 | 3729544 | -27139846 | 0.0717 | USDC | 14.5724 | chain stablecoin distribution context |
| BSC | BNB | chain_stablecoin_context | 12545858262 | -347575 | -59424015 | -131108820 | -0.0047 | USD1 | 13.9140 | chain stablecoin distribution context |
| Plasma | - | chain_stablecoin_context | 937867878 | -5593192 | 40785768 | -23622660 | 0.0455 | USDT | 12.0701 | chain stablecoin distribution context |
| MegaETH | - | chain_stablecoin_context | 437386654 | 1744329 | 15207139 | 32343728 | 0.0360 | USDe | 8.4019 | chain stablecoin distribution context |
| XDC | - | chain_stablecoin_context | 59916000 | -27061069 | -1909327 | -15505514 | -0.0309 | USDC | 6.3319 | chain stablecoin distribution context |
| Osmosis | - | chain_stablecoin_context | 22425157 | 80955 | -593713 | -1141361 | -0.0258 | USDC | 5.2106 | chain stablecoin distribution context |

## Interpretation

Stablecoin inflow can indicate deployable capital arriving on a chain; outflow can indicate risk-off, bridge withdrawal, or venue-specific liquidity stress. This still needs token mapping, venue coverage, bridge route checks, and forward labels.
