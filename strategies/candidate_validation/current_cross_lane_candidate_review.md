# Current Cross-Lane Candidate Review

This consolidates current candidate screens and first short-horizon labels. It is a triage board, not a deployable strategy ranking.

| asset | score | lanes | positive labels | negative labels | pending labels | note |
| --- | ---: | --- | --- | --- | --- | --- |
| WLD | 7.0571 | hl_candidate_label; okx_pressure; okx_liquidation | hl15=0.0197; okx_pressure15=0.0247; liq_cont15=0.0273 |  |  | first labels support follow-up |
| MEGA | 4.1464 | hl_candidate_label; on_chain_flow | hl15=0.0178; chain15=0.0025:HL:chain_flow_reversal_watch; chain15=0.0021:OKX:chain_flow_reversal_watch |  |  | first labels support follow-up |
| BTC | 3.3395 | okx_pressure; okx_liquidation; l2_imbalance_monitor; on_chain_flow | liq_cont15=0.0020; l2_imbalance15=0.0001; chain15=0.0005:HL:chain_flow_reversal_watch; chain15=0.0005:OKX:chain_flow_reversal_watch | okx_pressure15=-0.0021 |  | mixed evidence; isolate which source is real |
| XMR | 3.1004 | hl_candidate_label; sector_rotation | hl15=0.0111 | sector15=-0.0005:Privacy |  | mixed evidence; isolate which source is real |
| IP | 2.8166 | hl_candidate_label; okx_pressure | hl15=0.0160 | okx_pressure15=-0.0009 |  | mixed evidence; isolate which source is real |
| SOL | 2.7520 | okx_pressure; okx_liquidation; l2_imbalance_monitor; on_chain_flow | okx_pressure15=0.0031; liq_cont15=0.0017; chain15=0.0011:OKX:chain_flow_reversal_watch; chain15=0.0009:HL:chain_flow_reversal_watch | l2_imbalance15=-0.0033 |  | mixed evidence; isolate which source is real |
| ONDO | 2.6106 | okx_pressure; okx_liquidation; l2_imbalance_monitor; sector_rotation | liq_cont15=0.0020; sector15=0.0029:Binance Alpha Spotlight | okx_pressure15=-0.0029; l2_imbalance15=-0.0046 |  | mixed evidence; isolate which source is real |
| ALLO | 2.5965 | okx_pressure; okx_liquidation | liq_cont15=0.0198 | okx_pressure15=-0.0078 |  | mixed evidence; isolate which source is real |
| HYPE | 2.4788 | okx_pressure; okx_liquidation; on_chain_flow | liq_cont15=0.0024; chain15=0.0043:HL:chain_flow_reversal_watch; chain15=0.0041:OKX:chain_flow_reversal_watch | okx_pressure15=-0.0007 |  | mixed evidence; isolate which source is real |
| JTO | 2.4579 | okx_pressure; okx_liquidation; l2_imbalance_monitor | liq_cont15=0.0003; l2_imbalance15=0.0125 | okx_pressure15=-0.0010 |  | mixed evidence; isolate which source is real |
| XPL | 2.4493 | okx_pressure; l2_imbalance_monitor; sector_rotation | l2_imbalance15=0.0030; sector15=0.0035:Echo Launchpad | okx_pressure15=-0.0037 |  | mixed evidence; isolate which source is real |
| HOME | 2.3356 | okx_pressure; okx_liquidation | okx_pressure15=0.0070 | liq_cont15=-0.0074 |  | mixed evidence; isolate which source is real |
| H | 2.2846 | okx_pressure; okx_liquidation | liq_cont15=0.0131 | okx_pressure15=-0.0005 |  | mixed evidence; isolate which source is real |
| ZORA | 2.2743 | hl_candidate_label | hl15=0.0055 |  |  | first labels support follow-up |
| KAITO | 2.1882 | hl_candidate_label | hl15=0.0038 |  |  | first labels support follow-up |
| AIXBT | 2.1603 | hl_candidate_label | hl15=0.0032 |  |  | first labels support follow-up |
| APEX | 2.1524 | hl_candidate_label | hl15=0.0030 |  |  | first labels support follow-up |
| BSV | 2.0874 | hl_candidate_label | hl15=0.0017 |  |  | first labels support follow-up |
| ETH | 2.0858 | okx_pressure; okx_liquidation; l2_imbalance_monitor; on_chain_flow | okx_pressure15=0.0007; liq_cont15=0.0011; l2_imbalance15=0.0010; chain15=0.0003:OKX:chain_flow_reversal_watch; chain15=0.0003:HL:chain_flow_reversal_watch |  |  | first labels support follow-up |
| SAGA | 2.0365 | hl_candidate_label | hl15=0.0007 |  |  | first labels support follow-up |
| ZRO | 2.0173 | hl_candidate_label; okx_pressure | hl15=0.0013 | okx_pressure15=-0.0011 |  | mixed evidence; isolate which source is real |
| PUMP | 1.9792 | okx_pressure; okx_liquidation; sector_rotation | liq_cont15=0.0020; sector15=0.0027:Launchpad | okx_pressure15=-0.0020 |  | mixed evidence; isolate which source is real |
| XLM | 1.8742 | okx_pressure; okx_liquidation; l2_imbalance_monitor; on_chain_flow | okx_pressure15=0.0010; l2_imbalance15=0.0122 | liq_cont15=-0.0015; chain15=-0.0034:OKX:chain_outflow_stress_watch; chain15=-0.0036:HL:chain_outflow_stress_watch |  | mixed evidence; isolate which source is real |
| PEPE | 1.7269 | okx_pressure; okx_liquidation | okx_pressure15=0.0040; liq_cont15=0.0033 |  |  | first labels support follow-up |
| ADA | 1.6544 | okx_pressure; l2_imbalance_monitor; on_chain_flow | l2_imbalance15=0.0015; chain15=0.0016:HL:chain_flow_reversal_watch; chain15=0.0006:OKX:chain_flow_reversal_watch | okx_pressure15=0.0000 |  | mixed evidence; isolate which source is real |

## Interpretation

Higher score means more current evidence survived a short label or appeared in multiple lanes. A negative label does not kill a candidate if another PnL component, such as funding, is still unmodeled.
