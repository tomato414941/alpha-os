# Current Chain TVL Flow Venue Coverage

This checks whether chain TVL flow candidates have public perp venues. It does not validate fees, fills, or whether TVL accounting is stale.

| chain | token | action | week % | day % | HL | OKX | venues | followup |
| --- | --- | --- | ---: | ---: | --- | --- | ---: | --- |
| Stellar | XLM | chain_outflow_stress_watch | -0.1647 | -0.0349 | True | True | 2 | label XLM short stress on covered venues |
| Polygon | POL | chain_outflow_stress_watch | -0.1025 | -0.0120 | True | True | 2 | label POL short stress on covered venues |
| Cardano | ADA | chain_flow_reversal_watch | -0.2825 | 0.0132 | True | True | 2 | label ADA rebound continuation on covered venues |
| Avalanche | AVAX | chain_flow_reversal_watch | -0.2080 | 0.0097 | True | True | 2 | label AVAX rebound continuation on covered venues |
| MegaETH | MEGA | chain_flow_reversal_watch | -0.1964 | 0.0105 | True | True | 2 | label MEGA rebound continuation on covered venues |
| Berachain | BERA | chain_flow_reversal_watch | -0.1687 | 0.0077 | True | True | 2 | label BERA rebound continuation on covered venues |
| Stacks | STX | chain_flow_reversal_watch | -0.1543 | 0.0279 | True | True | 2 | label STX rebound continuation on covered venues |
| Arbitrum | ARB | chain_flow_reversal_watch | -0.1448 | 0.0362 | True | True | 2 | label ARB rebound continuation on covered venues |
| Sui | SUI | chain_flow_reversal_watch | -0.1435 | 0.0160 | True | True | 2 | label SUI rebound continuation on covered venues |
| Bitcoin | BTC | chain_flow_reversal_watch | -0.1384 | 0.0357 | True | True | 2 | label BTC rebound continuation on covered venues |
| Movement | MOVE | chain_flow_reversal_watch | -0.1360 | 0.0457 | True | True | 2 | label MOVE rebound continuation on covered venues |
| Sei | SEI | chain_flow_reversal_watch | -0.1143 | 0.0208 | True | True | 2 | label SEI rebound continuation on covered venues |
| Aptos | APT | chain_flow_reversal_watch | -0.1097 | 0.0213 | True | True | 2 | label APT rebound continuation on covered venues |
| Ethereum | ETH | chain_flow_reversal_watch | -0.1022 | 0.0387 | True | True | 2 | label ETH rebound continuation on covered venues |
| BSC | BNB | chain_flow_reversal_watch | -0.0986 | 0.0229 | True | True | 2 | label BNB rebound continuation on covered venues |
| Solana | SOL | chain_flow_reversal_watch | -0.0951 | 0.0360 | True | True | 2 | label SOL rebound continuation on covered venues |
| Monad | MON | chain_flow_reversal_watch | -0.0948 | 0.0170 | True | True | 2 | label MON rebound continuation on covered venues |
| OP Mainnet | OP | chain_flow_reversal_watch | -0.0928 | 0.0418 | True | True | 2 | label OP rebound continuation on covered venues |
| Hyperliquid L1 | HYPE | chain_flow_reversal_watch | -0.0922 | 0.0760 | True | True | 2 | label HYPE rebound continuation on covered venues |
| Near | NEAR | chain_flow_reversal_watch | -0.0827 | 0.1131 | True | True | 2 | label NEAR rebound continuation on covered venues |
| TON | TON | chain_flow_reversal_watch | -0.0676 | 0.0078 | True | True | 2 | label TON rebound continuation on covered venues |
| Starknet | STRK | chain_flow_reversal_watch | -0.0574 | 0.0098 | True | True | 2 | label STRK rebound continuation on covered venues |
| Plasma | XPL | chain_flow_context | -0.0079 | 0.0194 | True | True | 2 | keep XPL as context |
| Mantle | MNT | chain_outflow_stress_watch | -0.3240 | -0.0037 | True | False | 1 | label MNT short stress on covered venues |
| Katana | KAT | chain_flow_reversal_watch | -0.2822 | 0.0195 | False | True | 1 | label KAT rebound continuation on covered venues |
| Cronos | CRO | chain_flow_context | -0.0494 | 0.0153 | False | True | 1 | keep CRO as context |
| ENI | ENI | chain_inflow_momentum_watch | 0.2319 | 0.0026 | False | False | 0 | keep as context until a perp venue exists |
| Provenance | HASH | chain_inflow_momentum_watch | 0.0535 | 0.0036 | False | False | 0 | keep as context until a perp venue exists |
| Flare | FLR | chain_flow_reversal_watch | -0.1821 | 0.0372 | False | False | 0 | keep as context until a perp venue exists |
| Hydration | HDX | chain_flow_reversal_watch | -0.1799 | 0.0131 | False | False | 0 | keep as context until a perp venue exists |
