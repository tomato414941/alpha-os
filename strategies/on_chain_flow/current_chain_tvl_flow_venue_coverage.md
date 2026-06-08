# Current Chain TVL Flow Venue Coverage

This checks whether chain TVL flow candidates have public perp venues. It does not validate fees, fills, or whether TVL accounting is stale.

| chain | token | action | week % | day % | HL | OKX | venues | followup |
| --- | --- | --- | ---: | ---: | --- | --- | ---: | --- |
| Stellar | XLM | chain_outflow_stress_watch | -0.1656 | -0.0359 | True | True | 2 | label XLM short stress on covered venues |
| Polygon | POL | chain_outflow_stress_watch | -0.0943 | -0.0029 | True | True | 2 | label POL short stress on covered venues |
| Cardano | ADA | chain_flow_reversal_watch | -0.2601 | 0.0449 | True | True | 2 | label ADA rebound continuation on covered venues |
| Avalanche | AVAX | chain_flow_reversal_watch | -0.2028 | 0.0164 | True | True | 2 | label AVAX rebound continuation on covered venues |
| MegaETH | MEGA | chain_flow_reversal_watch | -0.1793 | 0.0319 | True | True | 2 | label MEGA rebound continuation on covered venues |
| Sui | SUI | chain_flow_reversal_watch | -0.1564 | 0.0007 | True | True | 2 | label SUI rebound continuation on covered venues |
| Stacks | STX | chain_flow_reversal_watch | -0.1516 | 0.0321 | True | True | 2 | label STX rebound continuation on covered venues |
| Berachain | BERA | chain_flow_reversal_watch | -0.1464 | 0.0347 | True | True | 2 | label BERA rebound continuation on covered venues |
| Arbitrum | ARB | chain_flow_reversal_watch | -0.1433 | 0.0381 | True | True | 2 | label ARB rebound continuation on covered venues |
| Bitcoin | BTC | chain_flow_reversal_watch | -0.1315 | 0.0440 | True | True | 2 | label BTC rebound continuation on covered venues |
| Movement | MOVE | chain_flow_reversal_watch | -0.1256 | 0.0583 | True | True | 2 | label MOVE rebound continuation on covered venues |
| Sei | SEI | chain_flow_reversal_watch | -0.1165 | 0.0182 | True | True | 2 | label SEI rebound continuation on covered venues |
| Aptos | APT | chain_flow_reversal_watch | -0.1085 | 0.0227 | True | True | 2 | label APT rebound continuation on covered venues |
| Ethereum | ETH | chain_flow_reversal_watch | -0.0969 | 0.0448 | True | True | 2 | label ETH rebound continuation on covered venues |
| OP Mainnet | OP | chain_flow_reversal_watch | -0.0925 | 0.0420 | True | True | 2 | label OP rebound continuation on covered venues |
| BSC | BNB | chain_flow_reversal_watch | -0.0903 | 0.0322 | True | True | 2 | label BNB rebound continuation on covered venues |
| Solana | SOL | chain_flow_reversal_watch | -0.0839 | 0.0487 | True | True | 2 | label SOL rebound continuation on covered venues |
| Monad | MON | chain_flow_reversal_watch | -0.0819 | 0.0314 | True | True | 2 | label MON rebound continuation on covered venues |
| Hyperliquid L1 | HYPE | chain_flow_reversal_watch | -0.0760 | 0.0951 | True | True | 2 | label HYPE rebound continuation on covered venues |
| Starknet | STRK | chain_flow_reversal_watch | -0.0564 | 0.0109 | True | True | 2 | label STRK rebound continuation on covered venues |
| Near | NEAR | chain_flow_context | -0.0425 | 0.1620 | True | True | 2 | keep NEAR as context |
| TON | TON | chain_flow_context | -0.0393 | 0.0384 | True | True | 2 | keep TON as context |
| Plasma | XPL | chain_flow_context | -0.0290 | -0.0023 | True | True | 2 | keep XPL as context |
| Mantle | MNT | chain_flow_reversal_watch | -0.3158 | 0.0084 | True | False | 1 | label MNT rebound continuation on covered venues |
| Katana | KAT | chain_flow_reversal_watch | -0.2596 | 0.0515 | False | True | 1 | label KAT rebound continuation on covered venues |
| Cronos | CRO | chain_flow_context | -0.0435 | 0.0217 | False | True | 1 | keep CRO as context |
| Provenance | HASH | chain_inflow_momentum_watch | 0.0623 | 0.0119 | False | False | 0 | keep as context until a perp venue exists |
| Mezo | MEZO | chain_flow_reversal_watch | -0.2590 | 0.0382 | False | False | 0 | keep as context until a perp venue exists |
| Hydration | HDX | chain_flow_reversal_watch | -0.1726 | 0.0220 | False | False | 0 | keep as context until a perp venue exists |
| ENI | ENI | chain_flow_reversal_watch | 0.1560 | -0.0592 | False | False | 0 | keep as context until a perp venue exists |
