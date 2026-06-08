# Current Chain Stablecoin Migration Forward Labels

This labels chain-level stablecoin migration against the mapped chain token. Positive directional return means the migration direction was right before costs and funding.

- total rows: `6`
- labeled 4h rows: `6`

| chain | token | migration | dir | week change | week % | dir 1h | dir 4h | dir 12h | label status | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Ethereum | ETH | chain_stablecoin_flow_reversal_watch | -1 | -2492385592 | -0.016626 | 0.01180894 | 0.02702544 |  | labeled_4h_pending_12h | wait for ETH 12h label before promotion |
| Arbitrum | ARB | chain_stablecoin_context_watch | -1 | -72009905 | -0.018918 | 0.01557259 | 0.02647341 |  | labeled_4h_pending_12h | wait for ARB 12h label before promotion |
| Hyperliquid L1 | HYPE | chain_stablecoin_flow_reversal_watch | -1 | -107642295 | -0.016516 | 0.03014701 | 0.02111262 |  | labeled_4h_pending_12h | wait for HYPE 12h label before promotion |
| Polygon | POL | paper_chain_stablecoin_outflow_watch | -1 | -129167095 | -0.035424 | 0.01031663 | 0.01461835 |  | labeled_4h_pending_12h | wait for POL 12h label before promotion |
| Tron | TRX | chain_stablecoin_flow_reversal_watch | -1 | -562996903 | -0.006254 | 0.00284421 | -0.00015291 |  | labeled_4h_pending_12h | wait for TRX 12h label before promotion |
| Solana | SOL | paper_chain_stablecoin_inflow_watch | 1 | 617116794 | 0.050732 | -0.01238970 | -0.02024000 |  | labeled_4h_pending_12h | wait for SOL 12h label before promotion |
