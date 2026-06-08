# Current Chain TVL Flow Venue Coverage

This checks whether chain TVL flow candidates have public perp venues. It does not validate fees, fills, or whether TVL accounting is stale.

| chain | token | action | week % | day % | HL | OKX | venues | followup |
| --- | --- | --- | ---: | ---: | --- | --- | ---: | --- |
| Stellar | XLM | chain_outflow_stress_watch | -0.1684 | -0.0391 | True | True | 2 | label XLM short stress on covered venues |
| Polygon | POL | chain_outflow_stress_watch | -0.0996 | -0.0087 | True | True | 2 | label POL short stress on covered venues |
| Cardano | ADA | chain_flow_reversal_watch | -0.2725 | 0.0273 | True | True | 2 | label ADA rebound continuation on covered venues |
| Avalanche | AVAX | chain_flow_reversal_watch | -0.2082 | 0.0095 | True | True | 2 | label AVAX rebound continuation on covered venues |
| MegaETH | MEGA | chain_flow_reversal_watch | -0.1833 | 0.0270 | True | True | 2 | label MEGA rebound continuation on covered venues |
| Berachain | BERA | chain_flow_reversal_watch | -0.1675 | 0.0091 | True | True | 2 | label BERA rebound continuation on covered venues |
| Stacks | STX | chain_flow_reversal_watch | -0.1559 | 0.0260 | True | True | 2 | label STX rebound continuation on covered venues |
| Sui | SUI | chain_flow_reversal_watch | -0.1501 | 0.0082 | True | True | 2 | label SUI rebound continuation on covered venues |
| Arbitrum | ARB | chain_flow_reversal_watch | -0.1408 | 0.0411 | True | True | 2 | label ARB rebound continuation on covered venues |
| Bitcoin | BTC | chain_flow_reversal_watch | -0.1361 | 0.0385 | True | True | 2 | label BTC rebound continuation on covered venues |
| Movement | MOVE | chain_flow_reversal_watch | -0.1262 | 0.0575 | True | True | 2 | label MOVE rebound continuation on covered venues |
| Sei | SEI | chain_flow_reversal_watch | -0.1163 | 0.0185 | True | True | 2 | label SEI rebound continuation on covered venues |
| Aptos | APT | chain_flow_reversal_watch | -0.1120 | 0.0186 | True | True | 2 | label APT rebound continuation on covered venues |
| Hyperliquid L1 | HYPE | chain_flow_reversal_watch | -0.1022 | 0.0641 | True | True | 2 | label HYPE rebound continuation on covered venues |
| Ethereum | ETH | chain_flow_reversal_watch | -0.1006 | 0.0405 | True | True | 2 | label ETH rebound continuation on covered venues |
| BSC | BNB | chain_flow_reversal_watch | -0.0970 | 0.0247 | True | True | 2 | label BNB rebound continuation on covered venues |
| Solana | SOL | chain_flow_reversal_watch | -0.0931 | 0.0383 | True | True | 2 | label SOL rebound continuation on covered venues |
| OP Mainnet | OP | chain_flow_reversal_watch | -0.0876 | 0.0477 | True | True | 2 | label OP rebound continuation on covered venues |
| Monad | MON | chain_flow_reversal_watch | -0.0859 | 0.0270 | True | True | 2 | label MON rebound continuation on covered venues |
| Near | NEAR | chain_flow_reversal_watch | -0.0685 | 0.1304 | True | True | 2 | label NEAR rebound continuation on covered venues |
| TON | TON | chain_flow_reversal_watch | -0.0634 | 0.0123 | True | True | 2 | label TON rebound continuation on covered venues |
| Starknet | STRK | chain_flow_reversal_watch | -0.0573 | 0.0100 | True | True | 2 | label STRK rebound continuation on covered venues |
| Plasma | XPL | chain_flow_context | 0.0123 | 0.0402 | True | True | 2 | keep XPL as context |
| Mantle | MNT | chain_flow_reversal_watch | -0.3198 | 0.0025 | True | False | 1 | label MNT rebound continuation on covered venues |
| Katana | KAT | chain_flow_reversal_watch | -0.2734 | 0.0319 | False | True | 1 | label KAT rebound continuation on covered venues |
| Cronos | CRO | chain_flow_context | -0.0462 | 0.0188 | False | True | 1 | keep CRO as context |
| ENI | ENI | chain_inflow_momentum_watch | 0.2320 | 0.0027 | False | False | 0 | keep as context until a perp venue exists |
| Provenance | HASH | chain_inflow_momentum_watch | 0.0553 | 0.0052 | False | False | 0 | keep as context until a perp venue exists |
| Mezo | MEZO | chain_flow_reversal_watch | -0.2609 | 0.0355 | False | False | 0 | keep as context until a perp venue exists |
| Hydration | HDX | chain_flow_reversal_watch | -0.1791 | 0.0141 | False | False | 0 | keep as context until a perp venue exists |
