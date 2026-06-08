# Current Chain TVL Flow Venue Coverage

This checks whether chain TVL flow candidates have public perp venues. It does not validate fees, fills, or whether TVL accounting is stale.

| chain | token | action | week % | day % | HL | OKX | venues | followup |
| --- | --- | --- | ---: | ---: | --- | --- | ---: | --- |
| MegaETH | MEGA | chain_outflow_stress_watch | -0.2078 | -0.0078 | True | True | 2 | label MEGA short stress on covered venues |
| Avalanche | AVAX | chain_outflow_stress_watch | -0.1979 | -0.0033 | True | True | 2 | label AVAX short stress on covered venues |
| Stellar | XLM | chain_outflow_stress_watch | -0.0740 | -0.0006 | True | True | 2 | label XLM short stress on covered venues |
| Cardano | ADA | chain_flow_reversal_watch | -0.2779 | 0.0180 | True | True | 2 | label ADA rebound continuation on covered venues |
| Movement | MOVE | chain_flow_reversal_watch | -0.1655 | 0.0355 | True | True | 2 | label MOVE rebound continuation on covered venues |
| Berachain | BERA | chain_flow_reversal_watch | -0.1649 | 0.0163 | True | True | 2 | label BERA rebound continuation on covered venues |
| Sui | SUI | chain_flow_reversal_watch | -0.1610 | 0.0309 | True | True | 2 | label SUI rebound continuation on covered venues |
| Stacks | STX | chain_flow_reversal_watch | -0.1576 | 0.0246 | True | True | 2 | label STX rebound continuation on covered venues |
| Arbitrum | ARB | chain_flow_reversal_watch | -0.1484 | 0.0290 | True | True | 2 | label ARB rebound continuation on covered venues |
| Bitcoin | BTC | chain_flow_reversal_watch | -0.1428 | 0.0271 | True | True | 2 | label BTC rebound continuation on covered venues |
| Ethereum | ETH | chain_flow_reversal_watch | -0.1210 | 0.0180 | True | True | 2 | label ETH rebound continuation on covered venues |
| Aptos | APT | chain_flow_reversal_watch | -0.1205 | 0.0071 | True | True | 2 | label APT rebound continuation on covered venues |
| Solana | SOL | chain_flow_reversal_watch | -0.1158 | 0.0118 | True | True | 2 | label SOL rebound continuation on covered venues |
| Near | NEAR | chain_flow_reversal_watch | -0.1094 | 0.0039 | True | True | 2 | label NEAR rebound continuation on covered venues |
| Sei | SEI | chain_flow_reversal_watch | -0.1075 | 0.0187 | True | True | 2 | label SEI rebound continuation on covered venues |
| BSC | BNB | chain_flow_reversal_watch | -0.1038 | 0.0303 | True | True | 2 | label BNB rebound continuation on covered venues |
| Hyperliquid L1 | HYPE | chain_flow_reversal_watch | -0.1024 | 0.0212 | True | True | 2 | label HYPE rebound continuation on covered venues |
| OP Mainnet | OP | chain_flow_reversal_watch | -0.0876 | 0.0405 | True | True | 2 | label OP rebound continuation on covered venues |
| Monad | MON | chain_flow_reversal_watch | -0.0850 | 0.0054 | True | True | 2 | label MON rebound continuation on covered venues |
| Polygon | POL | chain_flow_reversal_watch | -0.0831 | 0.0073 | True | True | 2 | label POL rebound continuation on covered venues |
| Starknet | STRK | chain_flow_reversal_watch | -0.0560 | 0.0033 | True | True | 2 | label STRK rebound continuation on covered venues |
| TON | TON | chain_flow_context | -0.0473 | 0.0556 | True | True | 2 | keep TON as context |
| Plasma | XPL | chain_flow_context | 0.0075 | 0.0154 | True | True | 2 | keep XPL as context |
| Mantle | MNT | chain_flow_reversal_watch | -0.3060 | 0.0237 | True | False | 1 | label MNT rebound continuation on covered venues |
| Katana | KAT | chain_flow_reversal_watch | -0.3000 | 0.0553 | False | True | 1 | label KAT rebound continuation on covered venues |
| Cronos | CRO | chain_flow_reversal_watch | -0.0550 | 0.0166 | False | True | 1 | label CRO rebound continuation on covered venues |
| ENI | ENI | chain_inflow_momentum_watch | 0.2330 | 0.0242 | False | False | 0 | keep as context until a perp venue exists |
| Flare | FLR | chain_flow_reversal_watch | -0.1820 | 0.0458 | False | False | 0 | keep as context until a perp venue exists |
| Hydration | HDX | chain_flow_reversal_watch | -0.1808 | 0.0055 | False | False | 0 | keep as context until a perp venue exists |
| Rootstock | RBTC | chain_flow_reversal_watch | -0.1356 | 0.0023 | False | False | 0 | keep as context until a perp venue exists |
