# Current Chain Stablecoin Migration

This aggregates DeFiLlama stablecoin chain-circulating data into chain-level liquidity migration. It is a capital-flow proxy, not a bridge-fill or trade instruction.

| chain | token | status | supply USD | day change | week change | month change | week % | top asset | score | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| Solana | SOL | paper_chain_stablecoin_inflow_watch | 13051020305 | 405138371 | 881166597 | 848460168 | 0.0724 | USDC | 89.4811 | large stablecoin inflow suggests deployable capital is arriving |
| Aptos | - | paper_chain_stablecoin_inflow_watch | 1139833199 | 48596300 | 108010874 | 58827063 | 0.1047 | USDT | 66.5404 | large stablecoin inflow suggests deployable capital is arriving |
| Hyperliquid L1 | HYPE | paper_chain_stablecoin_inflow_watch | 6792006091 | 399712874 | 235067902 | 1505674029 | 0.0359 | USDC | 65.7155 | large stablecoin inflow suggests deployable capital is arriving |
| Polygon | POL | paper_chain_stablecoin_outflow_watch | 3429274563 | -95414316 | -218594199 | -70889613 | -0.0599 | USDC | 64.3437 | large stablecoin outflow suggests capital is leaving |
| Ethereum | ETH | chain_stablecoin_flow_reversal_watch | 147318632826 | -83064781 | -2499493730 | -5785004689 | -0.0167 | USDT | 63.3367 | large weekly stablecoin flow has mixed daily confirmation |
| Base | - | paper_chain_stablecoin_inflow_watch | 4468690746 | 42734664 | 162405398 | -58231066 | 0.0377 | USDC | 60.1317 | large stablecoin inflow suggests deployable capital is arriving |
| Tron | TRX | chain_stablecoin_flow_reversal_watch | 89364903617 | -97201073 | -448381368 | -13052829 | -0.0050 | USDT | 58.4175 | large weekly stablecoin flow has mixed daily confirmation |
| Arbitrum | ARB | paper_chain_stablecoin_outflow_watch | 3647768698 | -66823909 | -121911731 | 121365938 | -0.0323 | PYUSD | 54.2114 | large stablecoin outflow suggests capital is leaving |
| Starknet | - | chain_stablecoin_context | 180594319 | 247177 | -81560659 | -135131650 | -0.3111 | USDC | 24.2586 | chain stablecoin distribution context |
| Stellar | - | chain_stablecoin_context | 274938261 | 5058343 | 70260435 | -1194636 | 0.3433 | USDC | 23.7880 | chain stablecoin distribution context |
| Flow | - | chain_stablecoin_context | 74357755 | 52835199 | 55196512 | 62790650 | 2.8806 | PYUSD | 22.8342 | chain stablecoin distribution context |
| Ink | - | chain_stablecoin_context | 181083538 | -5178109 | 26902030 | -10030245 | 0.1745 | USDC | 21.5262 | chain stablecoin distribution context |
| XDC | - | chain_stablecoin_context | 86962768 | -7975 | 25141936 | 11546739 | 0.4067 | USDC | 21.3441 | chain stablecoin distribution context |
| ZKsync Era | - | chain_stablecoin_context | 35822187 | -15340 | -6804829 | -7155434 | -0.1596 | USDC | 20.3761 | chain stablecoin distribution context |
| Cardano | - | chain_stablecoin_context | 20381986 | -148897 | -5985502 | 3301455 | -0.2270 | USDC | 20.3197 | chain stablecoin distribution context |
| World Chain | - | chain_stablecoin_context | 25490719 | 298317 | 4701424 | 6539383 | 0.2261 | USDC | 20.2606 | chain stablecoin distribution context |
| Algorand | - | chain_stablecoin_context | 56075063 | 2321518 | 4027032 | -26840110 | 0.0774 | USDC | 15.7317 | chain stablecoin distribution context |
| Sui | SUI | chain_stablecoin_context | 357924606 | -3442316 | 23370580 | -84989018 | 0.0699 | USDC | 15.4976 | chain stablecoin distribution context |
| Berachain | BERA | chain_stablecoin_context | 70221417 | 191163 | -5179279 | -7406863 | -0.0687 | USDe | 14.0672 | chain stablecoin distribution context |
| BSC | BNB | chain_stablecoin_context | 12546312198 | -1003253 | -60081922 | -131779895 | -0.0048 | USD1 | 13.9573 | chain stablecoin distribution context |
| Katana | - | chain_stablecoin_context | 23734266 | -123023 | -1711233 | -50901179 | -0.0673 | USDC | 13.5595 | chain stablecoin distribution context |
| Sei | - | chain_stablecoin_context | 55445690 | -3009284 | -3543354 | -2097687 | -0.0601 | USDC | 12.2462 | chain stablecoin distribution context |
| Hedera | - | chain_stablecoin_context | 40012707 | -1553633 | -2330808 | -6444109 | -0.0550 | USDC | 11.1656 | chain stablecoin distribution context |
| OP Mainnet | - | chain_stablecoin_context | 501191339 | -22932835 | -24519926 | -68003798 | -0.0466 | USDT | 11.0555 | chain stablecoin distribution context |
| MegaETH | - | chain_stablecoin_context | 440190917 | 4547489 | 18010149 | 35145895 | 0.0427 | USDe | 9.8727 | chain stablecoin distribution context |

## Interpretation

Stablecoin inflow can indicate deployable capital arriving on a chain; outflow can indicate risk-off, bridge withdrawal, or venue-specific liquidity stress. This still needs token mapping, venue coverage, bridge route checks, and forward labels.
