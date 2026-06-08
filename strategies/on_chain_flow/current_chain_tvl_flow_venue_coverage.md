# Current Chain TVL Flow Venue Coverage

This checks whether chain TVL flow candidates have public perp venues. It does not validate fees, fills, or whether TVL accounting is stale.

| chain | token | action | week % | day % | HL | OKX | venues | followup |
| --- | --- | --- | ---: | ---: | --- | --- | ---: | --- |
| Stellar | XLM | chain_outflow_stress_watch | -0.1572 | -0.0262 | True | True | 2 | label XLM short stress on covered venues |
| Polygon | POL | chain_outflow_stress_watch | -0.0961 | -0.0049 | True | True | 2 | label POL short stress on covered venues |
| Cardano | ADA | chain_flow_reversal_watch | -0.2664 | 0.0359 | True | True | 2 | label ADA rebound continuation on covered venues |
| Avalanche | AVAX | chain_flow_reversal_watch | -0.1987 | 0.0216 | True | True | 2 | label AVAX rebound continuation on covered venues |
| MegaETH | MEGA | chain_flow_reversal_watch | -0.1794 | 0.0319 | True | True | 2 | label MEGA rebound continuation on covered venues |
| Stacks | STX | chain_flow_reversal_watch | -0.1518 | 0.0318 | True | True | 2 | label STX rebound continuation on covered venues |
| Berachain | BERA | chain_flow_reversal_watch | -0.1486 | 0.0320 | True | True | 2 | label BERA rebound continuation on covered venues |
| Sui | SUI | chain_flow_reversal_watch | -0.1454 | 0.0137 | True | True | 2 | label SUI rebound continuation on covered venues |
| Arbitrum | ARB | chain_flow_reversal_watch | -0.1397 | 0.0424 | True | True | 2 | label ARB rebound continuation on covered venues |
| Bitcoin | BTC | chain_flow_reversal_watch | -0.1293 | 0.0466 | True | True | 2 | label BTC rebound continuation on covered venues |
| Movement | MOVE | chain_flow_reversal_watch | -0.1269 | 0.0567 | True | True | 2 | label MOVE rebound continuation on covered venues |
| Sei | SEI | chain_flow_reversal_watch | -0.1130 | 0.0222 | True | True | 2 | label SEI rebound continuation on covered venues |
| Aptos | APT | chain_flow_reversal_watch | -0.1098 | 0.0212 | True | True | 2 | label APT rebound continuation on covered venues |
| Ethereum | ETH | chain_flow_reversal_watch | -0.0942 | 0.0480 | True | True | 2 | label ETH rebound continuation on covered venues |
| BSC | BNB | chain_flow_reversal_watch | -0.0936 | 0.0285 | True | True | 2 | label BNB rebound continuation on covered venues |
| Solana | SOL | chain_flow_reversal_watch | -0.0858 | 0.0466 | True | True | 2 | label SOL rebound continuation on covered venues |
| OP Mainnet | OP | chain_flow_reversal_watch | -0.0846 | 0.0512 | True | True | 2 | label OP rebound continuation on covered venues |
| Hyperliquid L1 | HYPE | chain_flow_reversal_watch | -0.0830 | 0.0869 | True | True | 2 | label HYPE rebound continuation on covered venues |
| Monad | MON | chain_flow_reversal_watch | -0.0816 | 0.0317 | True | True | 2 | label MON rebound continuation on covered venues |
| TON | TON | chain_flow_reversal_watch | -0.0613 | 0.0145 | True | True | 2 | label TON rebound continuation on covered venues |
| Starknet | STRK | chain_flow_reversal_watch | -0.0570 | 0.0102 | True | True | 2 | label STRK rebound continuation on covered venues |
| Near | NEAR | chain_flow_reversal_watch | -0.0551 | 0.1467 | True | True | 2 | label NEAR rebound continuation on covered venues |
| Plasma | XPL | chain_flow_context | -0.0300 | -0.0033 | True | True | 2 | keep XPL as context |
| Mantle | MNT | chain_flow_reversal_watch | -0.3131 | 0.0123 | True | False | 1 | label MNT rebound continuation on covered venues |
| Katana | KAT | chain_flow_reversal_watch | -0.2647 | 0.0443 | False | True | 1 | label KAT rebound continuation on covered venues |
| Cronos | CRO | chain_flow_context | -0.0409 | 0.0245 | False | True | 1 | keep CRO as context |
| Provenance | HASH | chain_inflow_momentum_watch | 0.0584 | 0.0082 | False | False | 0 | keep as context until a perp venue exists |
| Mezo | MEZO | chain_flow_reversal_watch | -0.2618 | 0.0342 | False | False | 0 | keep as context until a perp venue exists |
| Hydration | HDX | chain_flow_reversal_watch | -0.1727 | 0.0220 | False | False | 0 | keep as context until a perp venue exists |
| ENI | ENI | chain_flow_reversal_watch | 0.1616 | -0.0546 | False | False | 0 | keep as context until a perp venue exists |
