# Current Chain Stablecoin Migration

This aggregates DeFiLlama stablecoin chain-circulating data into chain-level liquidity migration. It is a capital-flow proxy, not a bridge-fill or trade instruction.

| chain | token | status | supply USD | day change | week change | month change | week % | top asset | score | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| Solana | SOL | paper_chain_stablecoin_inflow_watch | 13040870218 | 395620937 | 871686128 | 838932552 | 0.0716 | USDC | 89.3261 | large stablecoin inflow suggests deployable capital is arriving |
| Ethereum | ETH | chain_stablecoin_flow_reversal_watch | 147314531955 | -72812450 | -2489037783 | -5774788517 | -0.0166 | USDT | 63.3231 | large weekly stablecoin flow has mixed daily confirmation |
| Polygon | POL | paper_chain_stablecoin_outflow_watch | 3441231198 | -83396172 | -206582703 | -58892298 | -0.0566 | USDC | 63.0968 | large stablecoin outflow suggests capital is leaving |
| Base | - | paper_chain_stablecoin_inflow_watch | 4474564956 | 48580050 | 168252202 | -52387052 | 0.0391 | USDC | 60.7014 | large stablecoin inflow suggests deployable capital is arriving |
| Tron | TRX | chain_stablecoin_flow_reversal_watch | 89336088958 | -112335713 | -463460203 | -28198355 | -0.0052 | USDT | 59.2052 | large weekly stablecoin flow has mixed daily confirmation |
| Aptos | - | chain_stablecoin_context_watch | 1131103632 | 40015993 | 99418657 | 50224137 | 0.0964 | USDT | 40.3751 | large chain has a material stablecoin distribution change |
| Hyperliquid L1 | HYPE | chain_stablecoin_flow_reversal_watch | 6453083534 | 60746914 | -103901679 | 1166717086 | -0.0158 | USDC | 39.8174 | large weekly stablecoin flow has mixed daily confirmation |
| Arbitrum | ARB | chain_stablecoin_context_watch | 3690122885 | -24361144 | -79467449 | 163820753 | -0.0211 | PYUSD | 26.8797 | large chain has a material stablecoin distribution change |
| Starknet | - | chain_stablecoin_context | 180493043 | 144262 | -81664564 | -135236256 | -0.3115 | USDC | 24.2637 | chain stablecoin distribution context |
| Stellar | - | chain_stablecoin_context | 271201429 | 1318228 | 66521114 | -4934827 | 0.3250 | USDC | 23.5973 | chain stablecoin distribution context |
| Ink | - | chain_stablecoin_context | 186851421 | 604844 | 32686847 | -4242787 | 0.2120 | USDC | 21.8212 | chain stablecoin distribution context |
| XDC | - | chain_stablecoin_context | 86971711 | -83 | 25150133 | 11554771 | 0.4068 | USDC | 21.3445 | chain stablecoin distribution context |
| ZKsync Era | - | chain_stablecoin_context | 35833771 | -3624 | -6793205 | -7143834 | -0.1594 | USDC | 20.3755 | chain stablecoin distribution context |
| Cardano | - | chain_stablecoin_context | 20365163 | -165921 | -6002591 | 3284484 | -0.2276 | USDC | 20.3205 | chain stablecoin distribution context |
| World Chain | - | chain_stablecoin_context | 25419770 | 227062 | 4630222 | 6468204 | 0.2227 | USDC | 20.2569 | chain stablecoin distribution context |
| Katana | - | chain_stablecoin_context | 22507646 | -1348584 | -2936723 | -52123341 | -0.1154 | USDC | 20.1693 | chain stablecoin distribution context |
| Flow | - | chain_stablecoin_context | 21424553 | -101621 | 2260146 | 9855735 | 0.1179 | PYUSD | 20.1344 | chain stablecoin distribution context |
| Sui | SUI | chain_stablecoin_context | 361651271 | 296040 | 27109530 | -81251525 | 0.0810 | USDC | 17.9241 | chain stablecoin distribution context |
| Algorand | - | chain_stablecoin_context | 56077839 | 2323778 | 4029313 | -26838205 | 0.0774 | USDC | 15.7405 | chain stablecoin distribution context |
| Berachain | BERA | chain_stablecoin_context | 70171099 | 145791 | -5224774 | -7452064 | -0.0693 | USDe | 14.1910 | chain stablecoin distribution context |
| BSC | BNB | chain_stablecoin_context | 12544493527 | -571379 | -59621202 | -131293155 | -0.0047 | USD1 | 13.9271 | chain stablecoin distribution context |
| Avalanche | AVAX | chain_stablecoin_context | 815850119 | -66401608 | -44967883 | -291060092 | -0.0522 | USDT | 13.5120 | chain stablecoin distribution context |
| Plasma | - | chain_stablecoin_context | 934329243 | -9107668 | 37269047 | -27137351 | 0.0415 | USDT | 11.1069 | chain stablecoin distribution context |
| MegaETH | - | chain_stablecoin_context | 438792777 | 3142966 | 16606083 | 33743328 | 0.0393 | USDe | 9.1358 | chain stablecoin distribution context |
| Osmosis | - | chain_stablecoin_context | 22506636 | 163764 | -510869 | -1058478 | -0.0222 | USDC | 4.4870 | chain stablecoin distribution context |

## Interpretation

Stablecoin inflow can indicate deployable capital arriving on a chain; outflow can indicate risk-off, bridge withdrawal, or venue-specific liquidity stress. This still needs token mapping, venue coverage, bridge route checks, and forward labels.
