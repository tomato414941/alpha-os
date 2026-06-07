# Current Cross-Lane Candidate Review

This consolidates current candidate screens and first short-horizon labels. It is a triage board, not a deployable strategy ranking.

| asset | score | lanes | positive labels | negative labels | pending labels | note |
| --- | ---: | --- | --- | --- | --- | --- |
| BEAT | 4.0603 | okx_pressure; okx_liquidation | liq_cont15=0.0240 |  | okx_pressure15 | first labels support follow-up |
| MEGA | 3.5234 | hl_candidate_label; okx_pressure; on_chain_flow | hl15=0.0178 | chain15=-0.0015:OKX:chain_flow_reversal_watch; chain15=-0.0022:HL:chain_flow_reversal_watch | okx_pressure15 | mixed evidence; isolate which source is real |
| BTC | 3.0730 | okx_pressure; okx_liquidation; l2_imbalance_monitor; on_chain_flow | liq_cont15=0.0003; l2_imbalance15=0.0001 | chain15=-0.0007:OKX:chain_flow_reversal_watch; chain15=-0.0008:HL:chain_flow_reversal_watch | okx_pressure15 | mixed evidence; isolate which source is real |
| WLD | 2.9989 | hl_candidate_label; okx_pressure; okx_liquidation | hl15=0.0197 |  | okx_pressure15; liq_cont15 | first labels support follow-up |
| XMR | 2.9735 | hl_candidate_label; sector_rotation | hl15=0.0111 | sector15=-0.0022:Privacy |  | mixed evidence; isolate which source is real |
| IP | 2.8057 | hl_candidate_label; okx_pressure | hl15=0.0160 |  | okx_pressure15 | first labels support follow-up |
| JTO | 2.7076 | okx_pressure; okx_liquidation; l2_imbalance_monitor | liq_cont15=0.0023; l2_imbalance15=0.0125 |  | okx_pressure15 | first labels support follow-up |
| XLM | 2.3776 | okx_pressure; okx_liquidation; l2_imbalance_monitor; on_chain_flow | l2_imbalance15=0.0122; chain15=0.0024:OKX:chain_flow_reversal_watch; chain15=0.0019:HL:chain_flow_reversal_watch | liq_cont15=-0.0039 | okx_pressure15 | mixed evidence; isolate which source is real |
| ZORA | 2.2743 | hl_candidate_label | hl15=0.0055 |  |  | first labels support follow-up |
| KAITO | 2.1882 | hl_candidate_label | hl15=0.0038 |  |  | first labels support follow-up |
| AIXBT | 2.1603 | hl_candidate_label | hl15=0.0032 |  |  | first labels support follow-up |
| APEX | 2.1524 | hl_candidate_label | hl15=0.0030 |  |  | first labels support follow-up |
| BSV | 2.0874 | hl_candidate_label | hl15=0.0017 |  |  | first labels support follow-up |
| ZRO | 2.0633 | hl_candidate_label | hl15=0.0013 |  |  | first labels support follow-up |
| XPL | 2.0497 | okx_pressure; l2_imbalance_monitor; sector_rotation; sector_perp_context | l2_imbalance15=0.0030 | sector15=-0.0023:Echo Launchpad; sector_perp15=-0.0023:Echo Launchpad | okx_pressure15 | mixed evidence; isolate which source is real |
| SAGA | 2.0365 | hl_candidate_label | hl15=0.0007 |  |  | first labels support follow-up |
| HOME | 2.0000 | okx_pressure; okx_liquidation |  |  | okx_pressure15; liq_cont15 | waiting for elapsed labels |
| ZEC | 1.5609 | okx_pressure; okx_liquidation; sector_rotation; sector_perp_context |  | sector15=-0.0019:Privacy; sector15=-0.0019:Zero Knowledge (ZK); sector_perp15=-0.0019:Privacy; sector_perp15=-0.0019:Zero Knowledge (ZK) | okx_pressure15; liq_cont15 | current short labels are weak |
| OPN | 1.5173 | okx_pressure; okx_liquidation | liq_cont15=0.0047 |  | okx_pressure15 | first labels support follow-up |
| POL | 1.4238 | sector_rotation; on_chain_flow | sector15=0.0035:Zero Knowledge (ZK); chain15=0.0035:HL:chain_flow_reversal_watch; chain15=0.0013:OKX:chain_flow_reversal_watch |  |  | first labels support follow-up |
| H | 1.2848 | okx_pressure; okx_liquidation | liq_cont15=0.0026 |  | okx_pressure15 | first labels support follow-up |
| ADA | 1.2549 | okx_pressure; okx_liquidation; l2_imbalance_monitor; on_chain_flow | l2_imbalance15=0.0015 | liq_cont15=-0.0006; chain15=-0.0012:HL:chain_flow_reversal_watch; chain15=-0.0012:OKX:chain_flow_reversal_watch | okx_pressure15 | mixed evidence; isolate which source is real |
| AVAX | 1.0343 | okx_pressure; on_chain_flow | chain15=0.0010:OKX:chain_outflow_stress_watch; chain15=0.0010:HL:chain_outflow_stress_watch |  | okx_pressure15 | first labels support follow-up |
| ETH | 0.9723 | okx_pressure; okx_liquidation; l2_imbalance_monitor; on_chain_flow | l2_imbalance15=0.0010; chain15=0.0009:HL:chain_flow_reversal_watch; chain15=0.0009:OKX:chain_flow_reversal_watch | liq_cont15=-0.0014 | okx_pressure15 | mixed evidence; isolate which source is real |
| SUI | 0.9520 | okx_pressure; okx_liquidation; l2_imbalance_monitor; on_chain_flow | chain15=0.0003:HL:chain_flow_reversal_watch; chain15=0.0003:OKX:chain_flow_reversal_watch | liq_cont15=-0.0004; l2_imbalance15=-0.0019 | okx_pressure15 | mixed evidence; isolate which source is real |

## Interpretation

Higher score means more current evidence survived a short label or appeared in multiple lanes. A negative label does not kill a candidate if another PnL component, such as funding, is still unmodeled.
