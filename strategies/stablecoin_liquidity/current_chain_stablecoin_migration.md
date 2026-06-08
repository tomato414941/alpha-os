# Current Chain Stablecoin Migration

This aggregates DeFiLlama stablecoin chain-circulating data into chain-level liquidity migration. It is a capital-flow proxy, not a bridge-fill or trade instruction.

| chain | token | status | supply USD | day change | week change | month change | week % | top asset | score | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| Solana | SOL | paper_chain_stablecoin_inflow_watch | 12781392432 | 266823883 | 617116794 | 497514395 | 0.0507 | USDC | 85.1464 | large stablecoin inflow suggests deployable capital is arriving |
| Ethereum | ETH | chain_stablecoin_flow_reversal_watch | 147413588525 | -190361916 | -2492385592 | -7505248126 | -0.0166 | USDT | 63.3253 | large weekly stablecoin flow has mixed daily confirmation |
| Base | - | paper_chain_stablecoin_inflow_watch | 4496095552 | 91719006 | 192097717 | -72954501 | 0.0446 | USDC | 63.0275 | large stablecoin inflow suggests deployable capital is arriving |
| Tron | TRX | chain_stablecoin_flow_reversal_watch | 89455284502 | -289853420 | -562996903 | 591409270 | -0.0063 | USDT | 61.2509 | large weekly stablecoin flow has mixed daily confirmation |
| Polygon | POL | paper_chain_stablecoin_outflow_watch | 3517146466 | -16971975 | -129167095 | -12106658 | -0.0354 | USDC | 55.0603 | large stablecoin outflow suggests capital is leaving |
| Hyperliquid L1 | HYPE | chain_stablecoin_flow_reversal_watch | 6409924545 | -212332424 | -107642295 | 1155422672 | -0.0165 | USDC | 40.0952 | large weekly stablecoin flow has mixed daily confirmation |
| Aptos | - | chain_stablecoin_context_watch | 1091724733 | 51948962 | 71785468 | -4495710 | 0.0704 | USDT | 33.7574 | large chain has a material stablecoin distribution change |
| Arbitrum | ARB | chain_stablecoin_context_watch | 3734470779 | 23997554 | -72009905 | 158904422 | -0.0189 | PYUSD | 26.1185 | large chain has a material stablecoin distribution change |
| Starknet | - | chain_stablecoin_context | 180517447 | -834449 | -80906551 | -122739823 | -0.3095 | USDC | 24.2258 | chain stablecoin distribution context |
| Stellar | - | chain_stablecoin_context | 271124803 | 1298805 | 65005440 | -2312737 | 0.3154 | USDC | 23.5214 | chain stablecoin distribution context |
| Ink | - | chain_stablecoin_context | 187188685 | -811050 | 33554331 | -7137702 | 0.2184 | USDC | 21.8649 | chain stablecoin distribution context |
| XDC | - | chain_stablecoin_context | 86984460 | 27063298 | 25153825 | 21562159 | 0.4068 | USDC | 21.3447 | chain stablecoin distribution context |
| Plasma | - | chain_stablecoin_context | 957542513 | 25806617 | 71401887 | -2535626 | 0.0806 | USDT | 20.6429 | chain stablecoin distribution context |
| ZKsync Era | - | chain_stablecoin_context | 35839178 | -63615 | -6797195 | -7127002 | -0.1594 | USDC | 20.3757 | chain stablecoin distribution context |
| Katana | - | chain_stablecoin_context | 22202131 | -1882211 | -6988960 | -49440963 | -0.2394 | USDT | 20.3717 | chain stablecoin distribution context |
| Cardano | - | chain_stablecoin_context | 20331853 | -309925 | -6058081 | 3275309 | -0.2296 | USDC | 20.3232 | chain stablecoin distribution context |
| World Chain | - | chain_stablecoin_context | 25325149 | -104327 | 4246136 | 6961386 | 0.2014 | USDC | 20.2376 | chain stablecoin distribution context |
| Flow | - | chain_stablecoin_context | 21437156 | -98898 | 3174813 | 9978893 | 0.1738 | PYUSD | 20.1802 | chain stablecoin distribution context |
| Sui | SUI | chain_stablecoin_context | 362470852 | 1909831 | 28208480 | -73481715 | 0.0844 | USDC | 18.6509 | chain stablecoin distribution context |
| Avalanche | AVAX | chain_stablecoin_context | 819425890 | -83466809 | -56511385 | -267028792 | -0.0645 | USDT | 16.5481 | chain stablecoin distribution context |
| BSC | BNB | chain_stablecoin_context | 12548233089 | -65457 | -66842135 | -134171520 | -0.0053 | USD1 | 14.4018 | chain stablecoin distribution context |
| Algorand | - | chain_stablecoin_context | 55188172 | 1401966 | 3125006 | -27471319 | 0.0600 | USDC | 12.2161 | chain stablecoin distribution context |
| Flare | - | chain_stablecoin_context | 27485931 | 205109 | -1432340 | -124305 | -0.0495 | USDT | 10.0052 | chain stablecoin distribution context |
| Berachain | BERA | chain_stablecoin_context | 70124962 | 7789 | -3177427 | -9183304 | -0.0433 | USDe | 8.8984 | chain stablecoin distribution context |
| MegaETH | - | chain_stablecoin_context | 437612551 | 3052961 | 15789330 | 94372528 | 0.0374 | USDe | 8.7133 | chain stablecoin distribution context |

## Interpretation

Stablecoin inflow can indicate deployable capital arriving on a chain; outflow can indicate risk-off, bridge withdrawal, or venue-specific liquidity stress. This still needs token mapping, venue coverage, bridge route checks, and forward labels.
