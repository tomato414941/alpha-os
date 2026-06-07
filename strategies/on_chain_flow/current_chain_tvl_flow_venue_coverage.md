# Current Chain TVL Flow Venue Coverage

This checks whether chain TVL flow candidates have public perp venues. It does not validate fees, fills, or whether TVL accounting is stale.

| chain | token | action | week % | day % | HL | OKX | venues | followup |
| --- | --- | --- | ---: | ---: | --- | --- | ---: | --- |
| Avalanche | AVAX | chain_outflow_stress_watch | -0.2054 | -0.0126 | True | True | 2 | label AVAX short stress on covered venues |
| Near | NEAR | chain_outflow_stress_watch | -0.1317 | -0.0212 | True | True | 2 | label NEAR short stress on covered venues |
| Stellar | XLM | chain_outflow_stress_watch | -0.0772 | -0.0040 | True | True | 2 | label XLM short stress on covered venues |
| Starknet | STRK | chain_outflow_stress_watch | -0.0621 | -0.0032 | True | True | 2 | label STRK short stress on covered venues |
| Cardano | ADA | chain_flow_reversal_watch | -0.2861 | 0.0065 | True | True | 2 | label ADA rebound continuation on covered venues |
| MegaETH | MEGA | chain_flow_reversal_watch | -0.2000 | 0.0019 | True | True | 2 | label MEGA rebound continuation on covered venues |
| Movement | MOVE | chain_flow_reversal_watch | -0.1835 | 0.0133 | True | True | 2 | label MOVE rebound continuation on covered venues |
| Berachain | BERA | chain_flow_reversal_watch | -0.1724 | 0.0072 | True | True | 2 | label BERA rebound continuation on covered venues |
| Stacks | STX | chain_flow_reversal_watch | -0.1660 | 0.0144 | True | True | 2 | label STX rebound continuation on covered venues |
| Sui | SUI | chain_flow_reversal_watch | -0.1634 | 0.0279 | True | True | 2 | label SUI rebound continuation on covered venues |
| Arbitrum | ARB | chain_flow_reversal_watch | -0.1627 | 0.0116 | True | True | 2 | label ARB rebound continuation on covered venues |
| Bitcoin | BTC | chain_flow_reversal_watch | -0.1538 | 0.0138 | True | True | 2 | label BTC rebound continuation on covered venues |
| OP Mainnet | OP | chain_flow_reversal_watch | -0.1227 | 0.0005 | True | True | 2 | label OP rebound continuation on covered venues |
| Aptos | APT | chain_flow_reversal_watch | -0.1187 | 0.0092 | True | True | 2 | label APT rebound continuation on covered venues |
| Ethereum | ETH | chain_flow_reversal_watch | -0.1179 | 0.0216 | True | True | 2 | label ETH rebound continuation on covered venues |
| Solana | SOL | chain_flow_reversal_watch | -0.1100 | 0.0184 | True | True | 2 | label SOL rebound continuation on covered venues |
| BSC | BNB | chain_flow_reversal_watch | -0.1100 | 0.0232 | True | True | 2 | label BNB rebound continuation on covered venues |
| Hyperliquid L1 | HYPE | chain_flow_reversal_watch | -0.1097 | 0.0129 | True | True | 2 | label HYPE rebound continuation on covered venues |
| Sei | SEI | chain_flow_reversal_watch | -0.1091 | 0.0169 | True | True | 2 | label SEI rebound continuation on covered venues |
| Monad | MON | chain_flow_reversal_watch | -0.0827 | 0.0079 | True | True | 2 | label MON rebound continuation on covered venues |
| Polygon | POL | chain_flow_reversal_watch | -0.0754 | 0.0157 | True | True | 2 | label POL rebound continuation on covered venues |
| TON | TON | chain_flow_reversal_watch | -0.0519 | 0.0505 | True | True | 2 | label TON rebound continuation on covered venues |
| Plasma | XPL | chain_flow_context | -0.0015 | 0.0063 | True | True | 2 | keep XPL as context |
| Katana | KAT | chain_flow_reversal_watch | -0.3122 | 0.0369 | False | True | 1 | label KAT rebound continuation on covered venues |
| Mantle | MNT | chain_flow_reversal_watch | -0.3121 | 0.0148 | True | False | 1 | label MNT rebound continuation on covered venues |
| Cronos | CRO | chain_flow_reversal_watch | -0.0582 | 0.0132 | False | True | 1 | label CRO rebound continuation on covered venues |
| ENI | ENI | chain_inflow_momentum_watch | 0.2403 | 0.0302 | False | False | 0 | keep as context until a perp venue exists |
| Hydration | HDX | chain_outflow_stress_watch | -0.1859 | -0.0007 | False | False | 0 | keep as context until a perp venue exists |
| Rootstock | RBTC | chain_outflow_stress_watch | -0.1392 | -0.0019 | False | False | 0 | keep as context until a perp venue exists |
| Tron | TRON | chain_outflow_stress_watch | -0.1029 | -0.0002 | False | False | 0 | keep as context until a perp venue exists |
