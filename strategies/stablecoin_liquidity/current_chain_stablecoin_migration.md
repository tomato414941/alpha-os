# Current Chain Stablecoin Migration

This aggregates DeFiLlama stablecoin chain-circulating data into chain-level liquidity migration. It is a capital-flow proxy, not a bridge-fill or trade instruction.

| chain | token | status | supply USD | day change | week change | month change | week % | top asset | score | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| Solana | SOL | paper_chain_stablecoin_inflow_watch | 12837101805 | 311237101 | 661932980 | 542053640 | 0.0544 | USDC | 85.8735 | large stablecoin inflow suggests deployable capital is arriving |
| Ethereum | ETH | chain_stablecoin_flow_reversal_watch | 147467958840 | -203055726 | -2506069171 | -7522592656 | -0.0167 | USDT | 63.3420 | large weekly stablecoin flow has mixed daily confirmation |
| Tron | TRX | chain_stablecoin_flow_reversal_watch | 89454670020 | -291795757 | -564942864 | 589479832 | -0.0063 | USDT | 61.2552 | large weekly stablecoin flow has mixed daily confirmation |
| Base | - | paper_chain_stablecoin_inflow_watch | 4475398072 | 65559149 | 166066497 | -99327611 | 0.0385 | USDC | 60.4860 | large stablecoin inflow suggests deployable capital is arriving |
| Polygon | POL | paper_chain_stablecoin_outflow_watch | 3523699694 | -12976836 | -125314075 | -7929765 | -0.0343 | USDC | 54.6578 | large stablecoin outflow suggests capital is leaving |
| Hyperliquid L1 | HYPE | chain_stablecoin_flow_reversal_watch | 6413213700 | -217224012 | -112407859 | 1152266589 | -0.0172 | USDC | 40.4787 | large weekly stablecoin flow has mixed daily confirmation |
| Aptos | - | chain_stablecoin_context_watch | 1091488012 | 51525126 | 71311054 | -5105893 | 0.0699 | USDT | 33.6372 | large chain has a material stablecoin distribution change |
| Arbitrum | ARB | chain_stablecoin_context_watch | 3744138528 | 30618314 | -65440929 | 165720361 | -0.0172 | PYUSD | 25.4518 | large chain has a material stablecoin distribution change |
| Starknet | - | chain_stablecoin_context | 180673859 | -905523 | -81080277 | -122967462 | -0.3098 | USDC | 24.2347 | chain stablecoin distribution context |
| Stellar | - | chain_stablecoin_context | 271472343 | 1300461 | 65088760 | -2315711 | 0.3154 | USDC | 23.5259 | chain stablecoin distribution context |
| Ink | - | chain_stablecoin_context | 187809556 | -297652 | 34123059 | -6596061 | 0.2220 | USDC | 21.8940 | chain stablecoin distribution context |
| XDC | - | chain_stablecoin_context | 87095911 | 27097989 | 25186069 | 21589799 | 0.4068 | USDC | 21.3464 | chain stablecoin distribution context |
| Plasma | - | chain_stablecoin_context | 957943651 | 26177214 | 71771751 | -2167984 | 0.0810 | USDT | 20.7447 | chain stablecoin distribution context |
| ZKsync Era | - | chain_stablecoin_context | 35881319 | -63693 | -6805981 | -7136351 | -0.1594 | USDC | 20.3762 | chain stablecoin distribution context |
| Katana | - | chain_stablecoin_context | 22275027 | -1828689 | -6937952 | -49418955 | -0.2375 | USDT | 20.3692 | chain stablecoin distribution context |
| Cardano | - | chain_stablecoin_context | 20357624 | -310234 | -6065707 | 3279681 | -0.2296 | USDC | 20.3236 | chain stablecoin distribution context |
| World Chain | - | chain_stablecoin_context | 25343809 | -118264 | 4237776 | 6956507 | 0.2008 | USDC | 20.2372 | chain stablecoin distribution context |
| Flow | - | chain_stablecoin_context | 21443985 | -98055 | 3176058 | 9981030 | 0.1739 | PYUSD | 20.1802 | chain stablecoin distribution context |
| Sui | SUI | chain_stablecoin_context | 362860408 | 1910043 | 28245735 | -73575876 | 0.0844 | USDC | 18.6577 | chain stablecoin distribution context |
| Avalanche | AVAX | chain_stablecoin_context | 820372066 | -83113141 | -56123200 | -266821562 | -0.0640 | USDT | 16.4328 | chain stablecoin distribution context |
| BSC | BNB | chain_stablecoin_context | 12552179595 | -146514 | -67006647 | -134371275 | -0.0053 | USD1 | 14.4123 | chain stablecoin distribution context |
| Algorand | - | chain_stablecoin_context | 55257976 | 1403889 | 3129137 | -27506409 | 0.0600 | USDC | 12.2171 | chain stablecoin distribution context |
| Flare | - | chain_stablecoin_context | 27422572 | 136163 | -1500452 | -194765 | -0.0519 | USDT | 10.4779 | chain stablecoin distribution context |
| Berachain | BERA | chain_stablecoin_context | 70150402 | 17826 | -3168156 | -9177434 | -0.0432 | USDe | 8.8707 | chain stablecoin distribution context |
| MegaETH | - | chain_stablecoin_context | 437675780 | 3045698 | 15784399 | 94381435 | 0.0374 | USDe | 8.7096 | chain stablecoin distribution context |

## Interpretation

Stablecoin inflow can indicate deployable capital arriving on a chain; outflow can indicate risk-off, bridge withdrawal, or venue-specific liquidity stress. This still needs token mapping, venue coverage, bridge route checks, and forward labels.
