# Current Chain Stablecoin Migration

This aggregates DeFiLlama stablecoin chain-circulating data into chain-level liquidity migration. It is a capital-flow proxy, not a bridge-fill or trade instruction.

| chain | token | status | supply USD | day change | week change | month change | week % | top asset | score | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| Solana | SOL | paper_chain_stablecoin_inflow_watch | 12968974534 | 322071698 | 798085984 | 765510361 | 0.0656 | USDC | 88.1147 | large stablecoin inflow suggests deployable capital is arriving |
| Aptos | - | paper_chain_stablecoin_inflow_watch | 1145824962 | 54405342 | 113834259 | 64661297 | 0.1103 | USDT | 66.8375 | large stablecoin inflow suggests deployable capital is arriving |
| Ethereum | ETH | chain_stablecoin_flow_reversal_watch | 147467003770 | 49211181 | -2367417075 | -5653239322 | -0.0158 | USDT | 63.1600 | large weekly stablecoin flow has mixed daily confirmation |
| Polygon | POL | paper_chain_stablecoin_outflow_watch | 3449511410 | -75419397 | -198600917 | -50904629 | -0.0544 | USDC | 62.2674 | large stablecoin outflow suggests capital is leaving |
| Base | - | paper_chain_stablecoin_inflow_watch | 4483374750 | 57406072 | 177077294 | -43559986 | 0.0411 | USDC | 61.5613 | large stablecoin inflow suggests deployable capital is arriving |
| Tron | TRX | chain_stablecoin_flow_reversal_watch | 89380352815 | -98303951 | -449545785 | -14136881 | -0.0050 | USDT | 58.4782 | large weekly stablecoin flow has mixed daily confirmation |
| Arbitrum | ARB | paper_chain_stablecoin_outflow_watch | 3650470400 | -64314099 | -119397591 | 123881275 | -0.0317 | PYUSD | 53.9547 | large stablecoin outflow suggests capital is leaving |
| Hyperliquid L1 | HYPE | chain_stablecoin_flow_reversal_watch | 6698317132 | 305948454 | 141303949 | 1411916538 | 0.0216 | USDC | 43.0735 | large weekly stablecoin flow has mixed daily confirmation |
| Starknet | - | chain_stablecoin_context | 180704976 | 356318 | -81451909 | -135023098 | -0.3107 | USDC | 24.2533 | chain stablecoin distribution context |
| Stellar | - | chain_stablecoin_context | 274943546 | 5062369 | 70264766 | -1190639 | 0.3433 | USDC | 23.7882 | chain stablecoin distribution context |
| Flow | - | chain_stablecoin_context | 73919179 | 52396881 | 54758159 | 62352188 | 2.8578 | PYUSD | 22.8118 | chain stablecoin distribution context |
| Ink | - | chain_stablecoin_context | 181715593 | -4566044 | 27512621 | -9423239 | 0.1784 | USDC | 21.5573 | chain stablecoin distribution context |
| XDC | - | chain_stablecoin_context | 86963187 | -7969 | 25142059 | 11546798 | 0.4067 | USDC | 21.3441 | chain stablecoin distribution context |
| ZKsync Era | - | chain_stablecoin_context | 35822845 | -15341 | -6804851 | -7155437 | -0.1596 | USDC | 20.3761 | chain stablecoin distribution context |
| Cardano | - | chain_stablecoin_context | 20106055 | -424980 | -6261617 | 3025378 | -0.2375 | USDC | 20.3332 | chain stablecoin distribution context |
| World Chain | - | chain_stablecoin_context | 25742649 | 550130 | 4953257 | 6791225 | 0.2383 | USDC | 20.2734 | chain stablecoin distribution context |
| Algorand | - | chain_stablecoin_context | 56048947 | 2294999 | 4000521 | -26866765 | 0.0769 | USDC | 15.6284 | chain stablecoin distribution context |
| Sui | SUI | chain_stablecoin_context | 357286371 | -4079490 | 22733236 | -85626713 | 0.0680 | USDC | 15.0842 | chain stablecoin distribution context |
| Berachain | BERA | chain_stablecoin_context | 70127876 | 85772 | -5285286 | -7513536 | -0.0701 | USDe | 14.3513 | chain stablecoin distribution context |
| BSC | BNB | chain_stablecoin_context | 12548748110 | -1058611 | -60162055 | -131872913 | -0.0048 | USD1 | 13.9624 | chain stablecoin distribution context |
| Sei | - | chain_stablecoin_context | 55450212 | -3005437 | -3539572 | -2093887 | -0.0600 | USDC | 12.2331 | chain stablecoin distribution context |
| Katana | - | chain_stablecoin_context | 23908878 | 49991 | -1538325 | -50732828 | -0.0605 | USDC | 12.1912 | chain stablecoin distribution context |
| Hedera | - | chain_stablecoin_context | 40012893 | -1553641 | -2330819 | -6444139 | -0.0550 | USDC | 11.1656 | chain stablecoin distribution context |
| OP Mainnet | - | chain_stablecoin_context | 502692177 | -21466229 | -23051627 | -66538670 | -0.0438 | USDT | 10.4244 | chain stablecoin distribution context |
| MegaETH | - | chain_stablecoin_context | 440412104 | 4680729 | 18146132 | 35285455 | 0.0430 | USDe | 9.9424 | chain stablecoin distribution context |

## Interpretation

Stablecoin inflow can indicate deployable capital arriving on a chain; outflow can indicate risk-off, bridge withdrawal, or venue-specific liquidity stress. This still needs token mapping, venue coverage, bridge route checks, and forward labels.
