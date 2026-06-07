# Current Cross-Lane Candidate Review

This consolidates current candidate screens and first short-horizon labels. It is a triage board, not a deployable strategy ranking.

| asset | score | lanes | positive labels | negative labels | pending labels | note |
| --- | ---: | --- | --- | --- | --- | --- |
| MEGA | 3.7120 | hl_candidate_label; okx_pressure; on_chain_flow | hl15=0.0178; chain15=0.0002:OKX:chain_flow_reversal_watch | chain15=-0.0004:HL:chain_flow_reversal_watch | okx_pressure15 | mixed evidence; isolate which source is real |
| BTC | 3.4455 | okx_pressure; okx_liquidation; l2_imbalance_monitor; on_chain_flow | liq_cont15=0.0003; l2_imbalance15=0.0001; chain15=0.0014:HL:chain_flow_reversal_watch; chain15=0.0014:OKX:chain_flow_reversal_watch |  | okx_pressure15 | first labels support follow-up |
| XMR | 3.1004 | hl_candidate_label; sector_rotation | hl15=0.0111 | sector15=-0.0005:Privacy |  | mixed evidence; isolate which source is real |
| WLD | 2.9989 | hl_candidate_label; okx_pressure; okx_liquidation | hl15=0.0197 |  | okx_pressure15; liq_cont15 | first labels support follow-up |
| BEAT | 2.9463 | okx_pressure; okx_liquidation | liq_cont15=0.0089 |  | okx_pressure15 | first labels support follow-up |
| JTO | 2.8539 | okx_pressure; okx_liquidation; l2_imbalance_monitor | liq_cont15=0.0037; l2_imbalance15=0.0125 |  | okx_pressure15 | first labels support follow-up |
| IP | 2.8057 | hl_candidate_label; okx_pressure | hl15=0.0160 |  | okx_pressure15 | first labels support follow-up |
| XPL | 2.6353 | okx_pressure; l2_imbalance_monitor; sector_rotation | l2_imbalance15=0.0030; sector15=0.0035:Echo Launchpad |  | okx_pressure15 | first labels support follow-up |
| ZORA | 2.2743 | hl_candidate_label | hl15=0.0055 |  |  | first labels support follow-up |
| KAITO | 2.1882 | hl_candidate_label | hl15=0.0038 |  |  | first labels support follow-up |
| AIXBT | 2.1603 | hl_candidate_label | hl15=0.0032 |  |  | first labels support follow-up |
| APEX | 2.1524 | hl_candidate_label | hl15=0.0030 |  |  | first labels support follow-up |
| BSV | 2.0874 | hl_candidate_label | hl15=0.0017 |  |  | first labels support follow-up |
| ZRO | 2.0633 | hl_candidate_label | hl15=0.0013 |  |  | first labels support follow-up |
| SAGA | 2.0365 | hl_candidate_label | hl15=0.0007 |  |  | first labels support follow-up |
| HOME | 2.0000 | okx_pressure; okx_liquidation |  |  | okx_pressure15; liq_cont15 | waiting for elapsed labels |
| HYPE | 1.7992 | okx_pressure; okx_liquidation; on_chain_flow | chain15=0.0068:OKX:chain_flow_reversal_watch; chain15=0.0067:HL:chain_flow_reversal_watch |  | okx_pressure15; liq_cont15 | first labels support follow-up |
| OPN | 1.7521 | okx_pressure; okx_liquidation | liq_cont15=0.0070 |  | okx_pressure15 | first labels support follow-up |
| ADA | 1.7028 | okx_pressure; okx_liquidation; l2_imbalance_monitor; on_chain_flow | l2_imbalance15=0.0015; chain15=0.0018:HL:chain_flow_reversal_watch; chain15=0.0012:OKX:chain_flow_reversal_watch | liq_cont15=-0.0006 | okx_pressure15 | mixed evidence; isolate which source is real |
| SOL | 1.6908 | okx_pressure; l2_imbalance_monitor; on_chain_flow | chain15=0.0031:OKX:chain_flow_reversal_watch; chain15=0.0030:HL:chain_flow_reversal_watch | l2_imbalance15=-0.0033 | okx_pressure15 | mixed evidence; isolate which source is real |
| MON | 1.5690 | hl_candidate_label; okx_pressure; on_chain_flow | chain15=0.0064:OKX:chain_flow_reversal_watch; chain15=0.0056:HL:chain_flow_reversal_watch | hl15=-0.0029 | okx_pressure15 | mixed evidence; isolate which source is real |
| XLM | 1.3646 | okx_pressure; okx_liquidation; l2_imbalance_monitor; on_chain_flow | l2_imbalance15=0.0122 | liq_cont15=-0.0039; chain15=-0.0063:OKX:chain_outflow_stress_watch; chain15=-0.0066:HL:chain_outflow_stress_watch | okx_pressure15 | mixed evidence; isolate which source is real |
| OP | 1.3503 | hl_candidate_label; okx_pressure; on_chain_flow | chain15=0.0044:HL:chain_flow_reversal_watch; chain15=0.0042:OKX:chain_flow_reversal_watch | hl15=-0.0001 | okx_pressure15 | mixed evidence; isolate which source is real |
| H | 1.2848 | okx_pressure; okx_liquidation | liq_cont15=0.0026 |  | okx_pressure15 | first labels support follow-up |
| POL | 1.1988 | sector_rotation; on_chain_flow | sector15=0.0016:Zero Knowledge (ZK); chain15=0.0025:OKX:chain_flow_reversal_watch; chain15=0.0022:HL:chain_flow_reversal_watch |  |  | first labels support follow-up |

## Interpretation

Higher score means more current evidence survived a short label or appeared in multiple lanes. A negative label does not kill a candidate if another PnL component, such as funding, is still unmodeled.
